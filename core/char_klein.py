"""StoryGPT-V character fusion and mask attention objectives adapted to Klein."""
import torch
from torch import nn
from torch.nn import functional as F


class CharacterFusion(nn.Module):
    """Inject a visual feature at its named text token; preserve other tokens."""
    def __init__(self, text_dim=7680, visual_dim=2560, hidden_dim=512):
        super().__init__()
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.in_proj = nn.Linear(text_dim + visual_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, text_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, text, visual, indices):
        if text.ndim != 3 or visual.ndim != 3 or indices.ndim != 2 or visual.shape[:2] != indices.shape:
            raise ValueError('Malformed character fusion inputs')
        if text.shape[0] != indices.shape[0] or (indices.numel() and (int(indices.min()) < -1 or int(indices.max()) >= text.shape[1])):
            raise ValueError('Character token outside the current caption')
        if any(len(set(row[row >= 0].tolist())) != int((row >= 0).sum()) for row in indices):
            raise ValueError('Multiple visual references assigned to the same text token')
        safe_indices = indices.clamp_min(0)
        chosen = text.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, text.shape[-1]))
        fused = torch.cat([self.text_norm(chosen.float()), self.visual_norm(visual.float())], dim=-1)
        delta = self.out_proj(F.gelu(self.in_proj(fused))).to(text.dtype) * (indices >= 0).unsqueeze(-1)
        return text.scatter_add(1, safe_indices.unsqueeze(-1).expand_as(delta), delta)


def mask_attention_loss(probabilities, masks):
    """Mean non-character attention minus mean attention inside each character."""
    if probabilities.ndim != 4 or masks.ndim != 3:
        raise ValueError('Expected B,H,P,K attention and B,K,P masks')
    if probabilities.shape[0] != masks.shape[0] or probabilities.shape[-2:] != (masks.shape[-1], masks.shape[1]):
        raise ValueError('Attention and supervision refer to different spatial locations')
    if masks.shape[1] == 0:
        return probabilities.sum() * 0.
    inside = masks.transpose(1, 2).unsqueeze(1).float()
    outside = 1. - inside
    valid = (inside.sum(dim=2) > 1e-5) & (outside.sum(dim=2) > 1e-5)
    positive = (probabilities.float() * inside).sum(dim=2) / inside.sum(dim=2).clamp_min(1e-5)
    negative = (probabilities.float() * outside).sum(dim=2) / outside.sum(dim=2).clamp_min(1e-5)
    expanded = valid.expand(-1, probabilities.shape[1], -1)
    return ((negative - positive) * expanded).sum() / expanded.sum().clamp_min(1)


class AttentionSupervision:
    def __init__(self):
        self.indices = None
        self.masks = None
        self.losses = []

    def set_batch(self, indices, masks):
        self.indices, self.masks, self.losses = indices, masks, []

    def record(self, query, key, text_length):
        if self.indices is None or not self.indices.numel():
            return
        # Klein has joint attention. Normalize its image-to-text slice over text
        # keys to adapt Char-LDM's cross-attention supervision without allocating
        # a quadratic image-to-image attention matrix.
        target_query = query[:, text_length:text_length + self.masks.shape[-1]].transpose(1, 2)
        text_key = key[:, :text_length].transpose(1, 2)
        scores = (target_query.float() @ text_key.float().transpose(-1, -2)) * (query.shape[-1] ** -.5)
        probabilities = scores.softmax(dim=-1)
        locations = self.indices.clamp_min(0)[:, None, None, :].expand(-1, probabilities.shape[1], probabilities.shape[2], -1)
        masks = self.masks * (self.indices >= 0).unsqueeze(-1)
        self.losses.append(mask_attention_loss(probabilities.gather(-1, locations), masks))


class SupervisedFluxAttention:
    """Native double-stream attention with an additional training-only loss."""
    _attention_backend = None
    _parallel_config = None

    def __init__(self, supervision):
        self.supervision = supervision

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections, apply_rotary_emb, dispatch_attention_fn
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if encoder_hidden_states is None or attn.added_kv_proj_dim is None:
            raise ValueError('Mask supervision requires Klein double-stream attention')
        encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
        encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        self.supervision.record(query, key, encoder_hidden_states.shape[1])
        result = dispatch_attention_fn(query, key, value, attn_mask=attention_mask,
                                       backend=self._attention_backend, parallel_config=self._parallel_config)
        result = result.flatten(2, 3).to(query.dtype)
        encoded, image = result.split_with_sizes([encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1)
        return attn.to_out[1](attn.to_out[0](image)), attn.to_add_out(encoded)


def install_attention_supervision(transformer, supervision, layers=(1, 3)):
    for index in layers:
        transformer.transformer_blocks[index].attn.set_processor(SupervisedFluxAttention(supervision))


class CharKlein(nn.Module):
    def __init__(self, transformer, fusion=None):
        super().__init__()
        self.transformer = transformer
        self.fusion = CharacterFusion() if fusion is None else fusion
        self.supervision = AttentionSupervision()
        install_attention_supervision(self.transformer, self.supervision)

    def forward(self, target, noise, sigma, text, text_ids, image_ids, visual, indices, masks, foreground,
                mask_only=False, attention_weight=.1):
        condition = self.fusion(text, visual, indices)
        self.supervision.set_batch(indices, masks)
        noisy = ((1. - sigma[:, None, None]) * target.float() + sigma[:, None, None] * noise.float()).to(target.dtype)
        prediction = self.transformer(hidden_states=noisy, encoder_hidden_states=condition, timestep=sigma.to(target.dtype),
                                      img_ids=image_ids, txt_ids=text_ids, guidance=None, return_dict=False)[0]
        error = prediction.float() - (noise.float() - target.float())
        selected = torch.as_tensor(mask_only, device=error.device, dtype=torch.bool).reshape(-1, 1)
        weights = torch.where(selected, foreground, torch.ones_like(foreground))
        denoise = (error * weights.unsqueeze(-1)).square().mean()
        regularizer = torch.stack(self.supervision.losses).mean() if self.supervision.losses else condition.sum() * 0.
        return denoise + attention_weight * regularizer, denoise.detach(), regularizer.detach()
