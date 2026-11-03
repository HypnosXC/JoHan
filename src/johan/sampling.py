"""Current ViT-H/512 joint 2D–3D flow sampler (10-step sine schedule)."""
import torch
import numpy as np
from .hand_network import HandDenoiseTraj
from .bbox_utils import generate_bbox_mask_per_frame_gpu
_DINO_EXTRACTOR = None

@torch.inference_mode()
def sample_hand(net, batch, device, num_flow_steps=10):
    if num_flow_steps<1: raise ValueError('num_flow_steps must be positive')
    b,t=batch.rgb_frames.shape[:2]
    if t>128: raise ValueError('Split video into windows of at most 128 frames')
    if _DINO_EXTRACTOR is None: raise RuntimeError('Load DINO before sampling')
    rgb=batch.rgb_frames.permute(0,1,4,2,3).to(torch.bfloat16)
    mean=torch.tensor([.485,.456,.406],device=device,dtype=torch.bfloat16).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device,dtype=torch.bfloat16).view(1,3,1,1)
    frames=(rgb.mul_(1.0/255.0)-mean)/std
    feat=_DINO_EXTRACTOR(frames.reshape(-1,*frames.shape[2:]))
    feat=feat.reshape(b,t,*feat.shape[1:]).float()
    mask=generate_bbox_mask_per_frame_gpu(batch.joint_2d_norm_params_per_frame,batch.img_shape)
    side=batch.mano_side[:,None].expand(b,t,1)
    conds=HandDenoiseTraj(mano_betas=torch.zeros(b,t,10,device=device),
        mano_poses=torch.zeros(b,t,15,3,device=device),global_orientation=torch.zeros(b,t,3,device=device),
        global_translation=torch.zeros(b,t,3,device=device),mano_side=side)
    # Preserve the original sampler RNG order, including unused global-pose noise.
    torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    x=torch.randn((b,t,net.get_d_state()),device=device)
    uv=torch.randn((b,t,21,3),device=device)
    translation=torch.randn((b,t,3),device=device)
    orientation=torch.randn((b,t,9),device=device)
    schedule=1.-np.sin(np.linspace(0.,1.,num_flow_steps+1)*np.pi/2)
    confidence=None
    for current,nxt in zip(schedule[:-1],schedule[1:]):
        index=max(1,min(int((1.-current)*net.config.max_t),net.config.max_t))
        pred,p2,confidence,_,_=net(x,torch.full((b,),index,device=device),
            rel_palm_pose=conds.global_translation,project_output_rotmats=False,conds=conds,
            img_feat=feat,img_feat_crop=None,mask=None,cond_dropout_keep_mask=torch.ones(b,device=device),
            noisy_joint_2d=uv,clean_joint_2d=None,bbox_mask=mask,
            noisy_translation=translation,noisy_ori_mat=orientation,cond_confidence=None)
        pred=HandDenoiseTraj.unpack(pred,using_mat=True,mano_side=side).pack(using_mat=True)
        dt=current-nxt
        x=x+(pred-x)/max(current,1e-5)*dt
        uv=uv+(torch.cat((p2,confidence),dim=-1)-uv)/max(current,1e-5)*dt
    return HandDenoiseTraj.unpack(x,using_mat=True,mano_side=side),uv[...,:2],confidence
