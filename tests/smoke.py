"""CPU integration checks: package closure, geometry, and valid-run windows."""
import importlib
import pkgutil
import numpy as np
import torch
import johan
for module in pkgutil.walk_packages(johan.__path__, 'johan.'):
    importlib.import_module(module.name)
from johan.cli import make_batch, windows
from johan.geometry import denormalize_2d_keypoints_minmax
from johan.network import TransformerBlockConfig
assert __import__("dataclasses").is_dataclass(TransformerBlockConfig)
frames=[np.zeros((48,80,3),np.uint8) for _ in range(3)]
b=make_batch(frames,np.array([[10,5,30,20]]*3),1,[100,110,40,24],256,'cpu')
assert b.rgb_frames.shape==(1,3,256,256)
assert torch.allclose(b.intrinsics[0,0],torch.tensor([320.,110*256/48,128,128]))
uv=denormalize_2d_keypoints_minmax(torch.zeros(1,3,21,2),b.joint_2d_norm_params_per_frame)
assert torch.allclose(uv[0,0,0],torch.tensor([64.,12.5*256/48]))
assert list(windows(np.array([1,1,0,1,1,1,1],bool),3))==[(0,2,0),(3,6,3),(4,7,6)]
assert not any(name.startswith('egoallo') for name in __import__('sys').modules)
print('CPU_IMPORT_GEOMETRY_WINDOWS_PASS')

if len(__import__('sys').argv)>1:
    from johan.checkpoint import load_model
    from johan import sampling
    from pathlib import Path
    net,run=load_model(Path(__import__('sys').argv[1]),'cpu')
    class FixedFeatures:
        def __call__(self,frames):
            return torch.zeros(len(frames),1280,32,32)
    sampling._DINO_EXTRACTOR=FixedFeatures()
    batch=make_batch(frames[:2],np.array([[10,5,30,20]]*2),1,[100,110,40,24],512,'cpu')
    with torch.inference_mode():
        traj,uv,conf=sampling.sample_hand(net,batch,torch.device('cpu'),num_flow_steps=2)
    assert traj.mano_poses.shape==(1,2,15,3)
    assert uv.shape==(1,2,21,2) and conf.shape==(1,2,21,1)
    assert torch.isfinite(traj.mano_poses).all() and torch.isfinite(uv).all()
    print('VITH_145000_STRICT_LOAD_AND_SYNTHETIC_FEATURE_SAMPLING_PASS')
