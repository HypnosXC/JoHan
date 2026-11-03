"""Export inference weights and plain YAML; exclude optimizer/training state."""
import argparse
import json
import shutil
from pathlib import Path
import yaml
from johan.checkpoint import read_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('destination',type=Path,help='New release directory; must not exist')
    p.add_argument('--weights',choices=['raw','ema','both'],default='raw')
    a=p.parse_args()
    modes=['raw','ema'] if a.weights=='both' else [a.weights]
    names=[]
    for mode in modes:
        suffix='_ema' if mode=='ema' else ''
        names += [f'model{suffix}.safetensors',f'dino_finetuned{suffix}.pt']
    for name in names:
        if not (a.source/name).is_file(): raise FileNotFoundError(a.source/name)
    model=read_config(a.source.parent/'model_config.yaml')
    run=read_config(a.source.parent/'run_config.yaml')
    if run.get('dino_model_name')!='dinov3_vith16plus' or model.get('image_size')!=512:
        raise ValueError('Only current ViT-H/512 checkpoints are supported')
    keys=('dino_model_name','image_size','subseq_len','use_feature','crop_before_dino',
          'crop_global_feat','gt_bbox_square','gt_bbox_scale','bbox_mask_square','bbox_mask_scale')
    meta=json.loads((a.source/'train_meta.json').read_text()) if (a.source/'train_meta.json').exists() else {}
    if 'coord_range' in meta: model['coord_range']=meta['coord_range']
    a.destination.mkdir(parents=True,exist_ok=False)
    dest=a.destination/'checkpoint';dest.mkdir()
    for name in names: shutil.copy2(a.source/name,dest/name)
    (a.destination/'model_config.yaml').write_text(yaml.safe_dump(model,sort_keys=True))
    (a.destination/'run_config.yaml').write_text(yaml.safe_dump({k:run[k] for k in keys},sort_keys=True))
    (dest/'train_meta.json').write_text(json.dumps({'coord_range':model['coord_range']},indent=2)+'\n')
    print(dest)

if __name__=='__main__': main()
