"""Inference-only DINOv3 patch features."""
import torch
from torch import nn

class DinoFeatureExtractor(nn.Module):
    def __init__(self, repo, model_name, weights, device):
        super().__init__()
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.model = torch.hub.load(str(repo), model_name, source='local',
                                   pretrained=False)
        self.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        self.to(device=self.device, dtype=dtype).eval()
        self.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, frames):
        h, w = frames.shape[-2] // 16, frames.shape[-1] // 16
        chunk = max(1, min(64, 32768 // (h*w)))
        outputs = []
        for batch in frames.split(chunk):
            tokens = self.model.forward_features(batch.to(self.device, dtype=next(self.model.parameters()).dtype))['x_norm_patchtokens']
            outputs.append(tokens.reshape(-1,h,w,tokens.shape[-1]).permute(0,3,1,2).contiguous())
        return torch.cat(outputs)
