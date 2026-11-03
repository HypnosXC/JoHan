"""Read legacy dataclass YAML as plain data; never import its Python tags."""
from pathlib import Path
import dataclasses
import json
import yaml
from safetensors.torch import load_file
from .hand_network_perjoint import HandDenoiserPerJoint, HandDenoiserPerJointConfig

class ConfigLoader(yaml.SafeLoader):
    pass
ConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/object:',
    lambda loader, suffix, node: loader.construct_mapping(node, deep=True))
ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
    lambda loader, node: loader.construct_sequence(node, deep=True))

def read_config(path):
    with open(path) as f:
        result = yaml.load(f, Loader=ConfigLoader)
    if not isinstance(result, dict):
        raise ValueError(f'Expected a configuration mapping: {path}')
    return result

def load_model(checkpoint, device, weights='raw'):
    checkpoint = Path(checkpoint)
    model_path = checkpoint.parent / 'model_config.yaml'
    values = read_config(model_path)
    # This legacy body-conditioning flag never participates in the hand-side-only model.
    values.pop('include_canonicalized_cpf_rotation_in_cond', None)
    cls = HandDenoiserPerJointConfig
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values)-known
    if unknown:
        raise ValueError(f'Unsupported model config fields: {sorted(unknown)}')
    meta = checkpoint/'train_meta.json'
    if meta.exists() and 'coord_range' in values:
        values['coord_range'] = json.loads(meta.read_text()).get('coord_range',values['coord_range'])
    cfg = cls(**values)
    if not cfg.predict_2Dpose or getattr(cfg,'pure_3d',False) or getattr(cfg,'pose2d_state_only',False) or getattr(cfg,'predict_global_pose',False):
        raise ValueError('This release supports joint 2D–3D hand checkpoints with PnP global pose.')
    expected = dict(image_size=512, pose2d_patch_feat_channels=1280, d_latent=640,
        num_heads=10, trunk_rounds=6, time_embed='sinusoidal', adaln_zero=True,
        flow_time_convention='flow', using_mat=True, use_optimal_2d=False,
        pose2d_conf_in_state=True, pose2d_regression=False, use_perjoint_pose2d=True,
        cond_param='only_side', use_mmdit=False, crop_feat_for_3d=False)
    for key, value in expected.items():
        if getattr(cfg,key,None)!=value:
            raise ValueError(f'Only current ViT-H/512 supported: {key} must be {value!r}')
    run = read_config(checkpoint.parent/'run_config.yaml')
    for key, value in dict(dino_model_name='dinov3_vith16plus',image_size=512,
        subseq_len=128,use_feature='online_dino',crop_before_dino=False,
        crop_global_feat=False,gt_bbox_square=True,gt_bbox_scale=1.0,
        bbox_mask_square=False,bbox_mask_scale=1.0).items():
        if run.get(key)!=value:
            raise ValueError(f'Unsupported ViT-H deployment setting {key}: {run.get(key)!r}')
    net = HandDenoiserPerJoint(cfg)
    filename = 'model_ema.safetensors' if weights=='ema' else 'model.safetensors'
    net.load_state_dict(load_file(str(checkpoint/filename)), strict=True)
    return net.to(device).eval(), run
