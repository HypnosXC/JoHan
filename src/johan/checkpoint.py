"""Read legacy dataclass YAML as plain data; never import its Python tags."""
from pathlib import Path
import dataclasses
import json
import yaml
from safetensors.torch import load_file
from .hand_network import HandDenoiser, HandDenoiserConfig
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
    perjoint = 'HandDenoiserPerJointConfig' in model_path.read_text() or 'num_rounds' in values
    cls = HandDenoiserPerJointConfig if perjoint else HandDenoiserConfig
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
    net = (HandDenoiserPerJoint if perjoint else HandDenoiser)(cfg)
    filename = 'model_ema.safetensors' if weights=='ema' else 'model.safetensors'
    net.load_state_dict(load_file(str(checkpoint/filename)), strict=True)
    return net.to(device).eval(), read_config(checkpoint.parent/'run_config.yaml')
