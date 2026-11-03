"""Compare direct trained-weight loading with the former two-stage loader on CPU."""
import argparse
import gc
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from johan.backbone import DinoFeatureExtractor
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--trained',type=Path,required=True)
p.add_argument('--original',type=Path,required=True)
a=p.parse_args()
legacy=nn.Module()
legacy.model=torch.hub.load(str(a.repo),'dinov3_vith16plus',source='local',weights=str(a.original)).eval()
state=torch.load(a.trained,map_location='cpu',weights_only=True)
legacy.load_state_dict(state,strict=True)
del state
frames=torch.linspace(0,1,3*512*512).reshape(1,3,512,512)
with torch.inference_mode():
    reference=legacy.model.forward_features(frames)['x_norm_patchtokens'].reshape(1,32,32,1280).permute(0,3,1,2).contiguous()
del legacy
gc.collect()
with patch('torch.hub.load_state_dict_from_url',side_effect=AssertionError('pretrained download attempted')):
    direct=DinoFeatureExtractor(a.repo,'dinov3_vith16plus',a.trained,'cpu')
    actual=direct(frames)
torch.testing.assert_close(actual,reference,rtol=0,atol=0)
print('DIRECT_TRAINED_BACKBONE_EXACT_PARITY_PASS',flush=True)
