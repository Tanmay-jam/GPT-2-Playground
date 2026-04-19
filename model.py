import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from blocks import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward_with_internals(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        positions = torch.arange(seq_len, device=in_idx.device).unsqueeze(0)
        pos_embeds = self.pos_emb(positions)
        input_embeds = tok_embeds + pos_embeds
        dropped = self.drop_emb(input_embeds)

        internals = {
            'tokens': None,
            'token_ids': in_idx.squeeze(0).tolist(),
            'embeddings': input_embeds.detach().cpu(),
            'positional_embeddings': pos_embeds.detach().cpu(),
            'layer_outputs': [],
            'logits': None
        }

        x = dropped
        for block in self.trf_blocks:
            x = block(x)
            internals['layer_outputs'].append(x.detach().cpu())

        normed = self.final_norm(x)
        logits = self.out_head(normed)
        internals['logits'] = logits.detach().cpu()

        return internals


def load_model(cfg):
    model = GPTModel(cfg)
    local_path = "/app/weights/gpt_pytorch_weights.pth"
    if os.path.exists(local_path):
        weights_path = local_path
    else:
        weights_path = hf_hub_download(
            repo_id="taanmaay/GPT-2-124M-weights",
            filename="gpt_pytorch_weights.pth",
            repo_type="model"
        )
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model
