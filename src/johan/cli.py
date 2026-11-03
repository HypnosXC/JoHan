"""Video to camera-space hand trajectories, without dataset annotations."""
import argparse
import json
import os
from pathlib import Path

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dino-repo',type=Path,required=True)
    p.add_argument('--dino-weights',type=Path,required=True)
    p.add_argument('--mano-root',type=Path,required=True)
    p.add_argument('--detector',type=Path,required=True,help='WiLoR YOLO hand detector checkpoint')
    p.add_argument('--intrinsics',type=float,nargs=4,required=True,metavar=('FX','FY','CX','CY'))
    p.add_argument('--output',type=Path,default=Path('outputs/predictions.npz'))
    p.add_argument('--weights',choices=['raw','ema'],default='raw')
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--confidence',type=float,default=.3)
    p.add_argument('--max-frames',type=int,default=0,help='0 processes the entire video')
    return p

def make_batch(rgb, boxes, side, intrinsics, size, device):
    import cv2
    import numpy as np
    import torch
    from .batch import HandInput
    t=len(rgb); h,w=rgb[0].shape[:2]; sx,sy=size/w,size/h
    frames=torch.from_numpy(np.stack([cv2.resize(im,(size,size),interpolation=cv2.INTER_AREA) for im in rgb]))[None]
    bx=torch.as_tensor(boxes,dtype=torch.float32)*torch.tensor([sx,sy,sx,sy])
    k=torch.tensor(intrinsics,dtype=torch.float32)*torch.tensor([sx,sy,sx,sy])
    return HandInput(mano_betas=torch.zeros(1,10),mano_pose=torch.zeros(1,t,51),
        joint_2d=torch.zeros(1,t,21,2),mano_joint_3d=torch.zeros(1,t,21,3),
        rgb_frames=frames,intrinsics=k[None,None].expand(1,t,4).clone(),
        extrinsics=torch.zeros(1,3,4),mask=torch.ones(1,t,dtype=torch.bool),
        mano_side=torch.tensor([[float(side)]]),img_feature=torch.zeros(1),
        img_shape=torch.tensor([[size,size]]),joint_2d_norm_params_per_frame=bx[None],
        bbox_mask=torch.zeros(1,1,size,size)).to(device)

def windows(valid, length):
    import numpy as np
    edges=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int)))
    for start,end in zip(edges[::2],edges[1::2]):
        done=start
        for s in range(start,end,length):
            e=min(s+length,end)
            s=max(start,e-length)
            yield s,e,done
            done=e

def main():
    args=parser().parse_args()
    if args.steps<1 or args.max_frames<0 or not 0<args.confidence<=1:
        raise ValueError('steps must be positive, max-frames nonnegative, confidence in (0,1]')
    if args.intrinsics[0]<=0 or args.intrinsics[1]<=0:
        raise ValueError('Focal lengths must be positive')
    os.environ['JOHAN_MANO_ROOT']=str(args.mano_root.resolve())
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from .checkpoint import load_model
    from .backbone import DinoFeatureExtractor
    from . import sampling
    from .geometry import denormalize_2d_keypoints_minmax, solve_global_pose_pnp
    net,config=load_model(args.checkpoint,args.device,args.weights)
    if config.get('use_feature','online_dino')!='online_dino':
        raise ValueError('Only DINO checkpoints are supported in this release')
    size=int(config.get('image_size',256)); length=int(config.get('subseq_len',128))
    if size%16 or length<1:
        raise ValueError('Invalid checkpoint image size or sequence length')
    dino=DinoFeatureExtractor(args.dino_repo,'dinov3_vith16plus',args.dino_weights,args.device)
    suffix='_ema' if args.weights=='ema' else ''
    backbone=args.checkpoint/f'dino_finetuned{suffix}.pt'
    if not backbone.exists():
        raise FileNotFoundError(f'Matching exported DINO weights required: {backbone}')
    dino.load_state_dict(torch.load(backbone,map_location='cpu',weights_only=True),strict=True)
    sampling._DINO_EXTRACTOR=dino
    capture=cv2.VideoCapture(str(args.video)); fps=capture.get(cv2.CAP_PROP_FPS)
    rgb=[]
    try:
        while not args.max_frames or len(rgb)<args.max_frames:
            ok,frame=capture.read()
            if not ok: break
            rgb.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: capture.release()
    if not rgb: raise ValueError(f'No video frames decoded: {args.video}')
    if any(im.shape!=rgb[0].shape for im in rgb): raise ValueError('Video resolution changed')
    detector=YOLO(str(args.detector))
    boxes=np.full((len(rgb),2,4),np.nan,np.float32)
    for start in range(0,len(rgb),32):
        results=detector.predict([cv2.cvtColor(im,cv2.COLOR_RGB2BGR) for im in rgb[start:start+32]],
            conf=args.confidence,device=args.device,verbose=False)
        for offset,result in enumerate(results):
            for side in (0,1):
                candidates=result.boxes[result.boxes.cls==side]
                if len(candidates):
                    index=int(candidates.conf.argmax())
                    boxes[start+offset,side]=candidates.xyxy[index].cpu().numpy()
    # WiLoR detector labels: 0=left, 1=right. One person / one hand per side.
    valid=np.isfinite(boxes).all(-1)&(boxes[...,2]>boxes[...,0])&(boxes[...,3]>boxes[...,1])
    for side in (0,1):
        ix=valid[:,side]; b=boxes[ix,side]; center=(b[:,:2]+b[:,2:])/2; wh=b[:,2:]-b[:,:2]
        if config.get('gt_bbox_square',False): wh=np.repeat(wh.max(-1,keepdims=True),2,-1)
        wh*=float(config.get('gt_bbox_scale',1.0)); boxes[ix,side]=np.c_[center-wh/2,center+wh/2]
    outputs={key:np.full((len(rgb),2,*shape),np.nan,np.float32) for key,shape in
             {'mano_pose':(45,), 'mano_betas':(10,), 'global_orient':(3,), 'translation':(3,), 'joints_2d':(21,2)}.items()}
    for side in (0,1):
        for s,e,keep in windows(valid[:,side],length):
            batch=make_batch(rgb[s:e],boxes[s:e,side],side,args.intrinsics,size,args.device)
            with torch.inference_mode():
                traj,uv,conf=sampling.sample_hand(net,batch,torch.device(args.device),num_flow_steps=args.steps)
            uv=((uv+1)*(size*.5) if getattr(net.config,'pose2d_absolute',False) else
                denormalize_2d_keypoints_minmax(uv,batch.joint_2d_norm_params_per_frame))[0].cpu()
            uv*=torch.tensor([rgb[0].shape[1]/size,rgb[0].shape[0]/size])
            pose=traj.mano_poses[0].reshape(e-s,45).cpu(); betas=traj.mano_betas[0].reshape(e-s,10).cpu()
            orient,trans=solve_global_pose_pnp(pose,betas,uv,torch.tensor(args.intrinsics).expand(e-s,4),'right' if side else 'left')
            for key,value in zip(outputs,(pose,betas,orient,trans,uv)):
                arr=value.numpy()
                if not np.isfinite(arr).all(): raise RuntimeError(f'Nonfinite {key} in frames {s}:{e}')
                outputs[key][keep:e,side]=arr[keep-s:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**outputs,valid=valid,bboxes=boxes,intrinsics=np.asarray(args.intrinsics),fps=fps)
    args.output.with_suffix('.json').write_text(json.dumps({'coordinate_frame':'camera','units':'metres',
        'hand_axis':['left','right'],'checkpoint':str(args.checkpoint),'weights':args.weights,
        'frames':len(rgb),'valid_hands':int(valid.sum()),'flow_steps':args.steps},indent=2)+'\n')
    print(f'Saved {args.output}: {len(rgb)} frames, {valid.sum()} hand observations')

if __name__=='__main__': main()
