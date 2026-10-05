import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, List, Callable, Tuple
from dataclasses import dataclass, asdict
from einops import rearrange, repeat
from .utils import weighted_quantile, interpolate_quantiles

@dataclass
class UTP2Config:
    quantiles: list
    patch_size: int
    rope_percentage: float # https://arxiv.org/pdf/2410.06205
    max_input_patches: int
    max_output_patches: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    rope_theta: float = 10000.0
    dropout: float = 0.0

class UTP2(nn.Module):
    def __init__(self, config: UTP2Config):
        super().__init__()
        self.config = config
        self.reg_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.rotary_emb = RotaryEmbedding(config)
        self.ts_encoder = TSEncoder(config)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_layers)
        ])
        self.predict_head = QuantilePredictHead(config)
        self.instance_norm = InstanceNorm()
        self.patch = Patch(config.patch_size, config.patch_size)

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor, num_output_patches: int) -> torch.Tensor:
        """
        Args:
            context: torch.Tensor, (batch_size, context_length)
            context_mask: torch.Tensor, (batch_size, context_length), 1 for observed patches
            num_output_patches: int, the number of output patches
        Returns:
            outputs: torch.Tensor, shape (batch_size, future_length, num_quantiles)
        """
        bs, _ = context.shape
        # 1. Patch the context and future, and concatenate them
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context, context_mask)
        patched_future = self._prepare_patched_future(patched_context.shape[0], num_output_patches)
        patched_context_future = torch.cat([patched_context, patched_future], dim=1)

        # 2. Encode the patched context and future and add [REG] token
        h = self.ts_encoder(patched_context_future)
        num_input_patches = patched_context.shape[1]
        h = torch.cat([h[:, :num_input_patches], self.reg_token.expand(bs, -1, -1), h[:, num_input_patches:]], dim=1)

        # 3. Generate rotary embeddings
        position_ids = torch.arange(0, h.shape[1], dtype=torch.long, device=h.device)
        position_ids = position_ids.unsqueeze(0).expand(bs, -1) # (B, P+1)
        cos, sin = self.rotary_emb(h, position_ids)
        cos, sin = mask_rope_cos_sin(cos, sin, self.config.rope_percentage)

        # 4. Build attention mask
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones(bs, 1, dtype=torch.bool, device=attention_mask.device),
            torch.ones(bs, num_output_patches, dtype=torch.bool, device=attention_mask.device)
        ], dim=1)
        attention_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)  # (B, L, L)
        attention_mask = attention_mask.unsqueeze(1) # (B, 1, L, L)

        # 5. Apply transformer layers
        for layer in self.encoder_layers:
            h = layer(h, (cos, sin), attention_mask)

        # 6. Predict quantiles for the output patches
        outputs = self.predict_head(h[:, -num_output_patches:]) # (bs, num_output_patches, patch_size * num_quantiles)
        # Reshape to (bs, num_output_patches * patch_size, num_quantiles)
        outputs = outputs.reshape(bs, num_output_patches * self.config.patch_size, -1)

        # 7. Inverse instance norm
        loc, scale = loc_scale
        loc_scale = (loc.unsqueeze(-1), scale.unsqueeze(-1)) # (bs, 1, 1) expand to quantile prediction shape
        outputs = self.instance_norm.inverse(outputs, loc_scale)
        return outputs

    @staticmethod
    def _get_prob_mass_per_quantile_level(quantile_levels: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized probability masses for quantile levels using trapezoidal rule approximation.
        """
        assert quantile_levels.ndim == 1
        assert quantile_levels.min() > 0.0 and quantile_levels.max() < 1.0

        device = quantile_levels.device
        boundaries = torch.cat(
            [torch.tensor([0.0], device=device), quantile_levels, torch.tensor([1.0], device=device)]
        )
        prob_mass = (boundaries[2:] - boundaries[:-2]) / 2
        return prob_mass / prob_mass.sum()

    @torch.no_grad()
    def predict(self, context: Optional[Union[torch.Tensor, List[torch.Tensor]]], prediction_length: int = 0,
                unrolled_quantiles: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) -> torch.Tensor:
        """
        Args:
            context: torch.Tensor, shape (batch_size, input_length) or list of 1D torch.Tensor
            prediction_length: int, the length of output
            unrolled_quantiles: list, the quantiles to predict
        Returns:
            predictions: torch.Tensor, shape (batch_size, prediction_length, num_quantiles)
        
        If context series have different lengths, left-pad each one with NaNs to the maximum length among them.
        unrolled_quantiles: The set of quantiles to use when making long-horizon predictions; must be a subset of the model's default quantiles. These quantiles
        are appended to the historical context and input into the model autoregressively to generate long-horizon predictions. Note that the
        effective batch size increases by multiplying `len(unrolled_quantiles)` when making long-horizon predictions.
        """
        self.eval()
        # 1. Convert list of tensors to a single tensor if necessary
        if isinstance(context, list):
            # 1. Pad each tensor in the list to the maximum length
            max_len = max(tensor.shape[0] for tensor in context)
            batch_size = len(context)
            padded_context = torch.full(
                (batch_size, max_len), 
                float('nan'), 
                device=context[0].device, 
                dtype=context[0].dtype
            )
            for i, tensor in enumerate(context):
                padded_context[i, -tensor.shape[0]:] = tensor
            context = padded_context

        context_mask = (~torch.isnan(context)).float()

        # 2. Use rolling window to generate predictions
        patch_size = self.config.patch_size
        num_output_patches = (prediction_length + patch_size - 1) // patch_size
        
        # Predict first set of patches up to max_output_patches
        step_patches = min(num_output_patches, self.config.max_output_patches)
        predictions = []
        
        # First step prediction
        prediction = self(context, context_mask, step_patches) # (B, step_len, Q)
        predictions.append(prediction)
        
        remaining_patches = num_output_patches - step_patches
        
        if remaining_patches > 0:
            # Prepare for long horizon unrolling
            unrolled_quantiles_tensor = torch.tensor(unrolled_quantiles, device=context.device, dtype=torch.float32)
            
            # Compute sample weights
            unrolled_sample_weights = torch.outer(
                self._get_prob_mass_per_quantile_level(unrolled_quantiles_tensor),
                self._get_prob_mass_per_quantile_level(torch.tensor(self.config.quantiles, device=context.device, dtype=torch.float32)),
            )
            
            # Expand context and mask
            # context: (B, T) -> (B, N_unroll, T)
            context = repeat(context, "b t -> b q t", q=len(unrolled_quantiles))
            context_mask = repeat(context_mask, "b t -> b q t", q=len(unrolled_quantiles))
            
            while remaining_patches > 0:
                step_patches = min(remaining_patches, self.config.max_output_patches)
                
                # Interpolate prediction to unrolled quantiles
                # prediction: (B, H, Q)
                # prediction_unrolled: (B, H, N_unroll)
                prediction_unrolled = interpolate_quantiles(
                    query_quantile_levels=unrolled_quantiles_tensor,
                    original_quantile_levels=self.config.quantiles,
                    original_values=prediction, 
                )
                
                # Append to context
                # prediction_unrolled: (B, H, N_unroll) -> (B, N_unroll, H)
                prediction_unrolled = rearrange(prediction_unrolled, "b h q -> b q h")
                context = torch.cat([context, prediction_unrolled], dim=-1)
                
                # Update mask
                step_mask = torch.ones_like(prediction_unrolled)
                context_mask = torch.cat([context_mask, step_mask], dim=-1)
                
                # Predict next step
                # context: (B, N_unroll, T+H) -> (B*N_unroll, T+H)
                flattened_context = rearrange(context, "b q t -> (b q) t")
                flattened_mask = rearrange(context_mask, "b q t -> (b q) t")
                
                # step_pred: (B*N_unroll, step_H, Q)
                step_pred = self(flattened_context, flattened_mask, step_patches)
                
                # Aggregate predictions
                # step_pred: (B*N, H, Q) -> (B, N*Q, H) -> (B, H, N*Q)
                n_paths = len(unrolled_quantiles)
                step_pred_agg = rearrange(step_pred, "(b n) h q -> b h (n q)", n=n_paths)
                
                # weighted_quantile expects samples to be (..., num_samples)
                # weights: (N, Q) -> (N*Q)
                flat_weights = rearrange(unrolled_sample_weights, "n q -> (n q)")
                
                prediction = weighted_quantile(
                    query_quantile_levels=self.config.quantiles,
                    sample_weights=flat_weights,
                    samples=step_pred_agg
                )
                # prediction: (B, H, Q)
                
                predictions.append(prediction)
                remaining_patches -= step_patches
        
        predictions = torch.cat(predictions, dim=1)
        return predictions[:, :prediction_length]
        

    def _prepare_patched_context(self, context: torch.Tensor, context_mask: torch.Tensor):
        """
        Args:
            context: torch.Tensor, shape (batch_size, input_length)
            context_mask: torch.Tensor, shape (batch_size, input_length), 1 for observed patches
        Returns:
            patched_context: (batch_size, num_context_patches, patch_size * 2)
            attention_mask: (batch_size, num_context_patches), 1 for observed patches
            loc_scale: tuple of (batch_size, 1)
        """
        # 1. Truncate context if it's longer than model's context length
        _, li = context.shape
        if li > (self.config.max_input_patches * self.config.patch_size):
            context = context[:, -self.config.max_input_patches * self.config.patch_size:]
            context_mask = context_mask[:, -self.config.max_input_patches * self.config.patch_size:]

        # 2. Scaling
        context, loc_scale = self.instance_norm(context)

        # 3. Patching
        # patched_context: (batch_size, num_patches, patch_size)
        # patched_mask: (batch_size, num_patches, patch_size)
        patched_context = self.patch(context) # (batch_size, num_patches, patch_size)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
        
        # 4. attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0 # (batch_size, num_patches)

        # 5. Concatenate context and mask along the last dimension
        patched_context = torch.cat([patched_context, patched_mask], dim=-1) # (batch_size, num_patches, patch_size * 2)
        return patched_context, attention_mask, loc_scale
            
    def _prepare_patched_future(self, batch_size: int, num_output_patches: int):
        """
        Args:
            batch_size: int, the batch size
            num_output_patches: int, the number of output patches
        Returns:
            patched_future: torch.Tensor, shape (batch_size, num_output_patches, patch_size * 2)
        """
        assert num_output_patches <= self.config.max_output_patches, f"num_output_patches ({num_output_patches}) must be less than or equal to max_output_patches ({self.config.max_output_patches})"
        patched_future = torch.zeros(batch_size, num_output_patches, self.config.patch_size * 2, device=self.reg_token.device, dtype=torch.float32)
        return patched_future


    @classmethod
    def load_model(cls, path: str, map_location: str = 'cpu'):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        config = UTP2Config(**state_dict['config'])
        model = UTP2(config).to(map_location)
        model.load_state_dict(state_dict['model'], strict=True)
        return model

    @classmethod
    def save_model(cls, model: 'UTP2', path: str):
        state_dict = {
            'config': asdict(model.config),
            'model': model.state_dict(),
        }
        torch.save(state_dict, path)

    @classmethod
    def compute_loss(
        cls, predictions: torch.Tensor, targets: torch.Tensor, 
        targets_mask: torch.Tensor, quantiles: List[float]
        ):
        """
        Inputs:
            predictions: torch.Tensor, shape (B, ..., num_quantiles)
            targets: torch.Tensor, shape (B, ...)
            targets_mask: torch.Tensor, shape (B, ...)
            quantiles: list of float, the quantiles to compute loss
        Output:
            loss: torch.Tensor, scalar

        Compute the pinball loss
        """
        q = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
        # Reshape q to match predictions rank: (1, ..., 1, num_quantiles)
        view_shape = [1] * (predictions.ndim - 1) + [-1]
        q = q.view(*view_shape)
        
        targets = targets.unsqueeze(-1) # (B, ..., 1)
        targets_mask = targets_mask.unsqueeze(-1) # (B, ..., 1)
        
        errors = targets - predictions
        loss = torch.max((q - 1) * errors, q * errors) * targets_mask # (B, ..., num_quantiles)
        
        return loss.sum() / (targets_mask.sum().clamp(min=1.0) * len(quantiles))
        

# Modified transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, config: UTP2Config, device=None):
        super().__init__()
        self.config = config

        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

        # Pre-compute and cache cos/sin
        self.max_seq_len_cached = config.max_input_patches + config.max_output_patches + 16
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos() * self.attention_scaling, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * self.attention_scaling, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[UTP2Config] = None,
        device: Optional["torch.device"] = None,
    ) -> Tuple["torch.Tensor", float]:
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )
        return inv_freq, attention_factor

    def forward(self, x, position_ids):
        """
        Args:
            x: torch.Tensor, (batch_size, seq_len, hidden_size)
            position_ids: torch.Tensor, (batch_size, seq_len)
        Returns:
            cos: torch.Tensor, (batch_size, seq_len, hidden_size)
            sin: torch.Tensor, (batch_size, seq_len, hidden_size)
        """
        # Support dynamic resizing if needed (though unlikely with fixed config)
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len_cached:
            return self._forward_dynamic(x, position_ids)
            
        return (
            F.embedding(position_ids, self.cos_cached).to(dtype=x.dtype),
            F.embedding(position_ids, self.sin_cached).to(dtype=x.dtype)
        )

    def _forward_dynamic(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def mask_rope_cos_sin(cos, sin, rope_percentage):
    """
    Applies p-RoPE masking to the cosine and sine tensors.
    """
    if rope_percentage >= 1.0:
        return cos, sin
        
    cos = cos.clone()
    sin = sin.clone()
    
    head_dim = cos.shape[-1]
    half_dim = head_dim // 2
    keep_bands = int(half_dim * rope_percentage)
    
    # Mask low frequencies
    cos[..., keep_bands:half_dim] = 1.0
    sin[..., keep_bands:half_dim] = 0.0
    cos[..., (half_dim + keep_bands):] = 1.0
    sin[..., (half_dim + keep_bands):] = 0.0
    
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies rotary position embedding to query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Modified gated_attention/blob/main/modeling_qwen3.py
class GatedSdpaAttention(nn.Module):
    """Gated Scaled Dot-Product Attention"""
    def __init__(self, config: UTP2Config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, h: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: a tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: a tuple of (cos, sin) of shape (batch_size, seq_len, hidden_size)
            attention_mask: a tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        BN, L, E = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(BN, L, self.num_heads, -1) # (BN, L, H, D*2)
        q, gate_score = torch.split(q, [self.head_dim, self.head_dim], dim=-1)
        # q: (BN, L, H, D)
        # gate_score: (BN, L, H, D)
        
        q = q.transpose(1, 2) # (BN, H, L, D)
        k = k.view(BN, L, self.num_heads, -1).transpose(1, 2)
        v = v.view(BN, L, self.num_heads, -1).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.dropout
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output * torch.sigmoid(gate_score)

        attn_output = attn_output.reshape(BN, L, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: UTP2Config):
        super().__init__()
        self.self_attn = GatedSdpaAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, h: torch.Tensor, position_embedding: Tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor):
        """
        Args:
            h: a tensor of shape (batch_size, seq_len, hidden_size)
            position_embedding: a tuple of (cos, sin) of shape (batch_size, seq_len, hidden_size)
            attention_mask: a tensor of shape (batch_size, seq_len, seq_len)
        Returns:
            (batch_size, seq_len, hidden_size)
        """
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(h, position_embedding, attention_mask)
        h = self.dropout(h)
        h = h + residual

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = self.dropout(h)
        h = h + residual

        return h


class MLP(nn.Module):
    def __init__(self, config: UTP2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor of shape (batch_size, seq_len, hidden_size)
        Output:
            (batch_size, seq_len, hidden_size)
        """
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class TSEncoder(nn.Module):
    """A MLP encoder for time series data."""
    def __init__(self, config: UTP2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.patch_size * 2, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.patch_size * 2, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor of shape (batch_size, num_patches, patch_size * 2)
        Output:
            (batch_size, num_patches, patch_size * 2)
        """
        h = torch.sigmoid(self.gate_proj(x)) * self.up_proj(x)
        h = self.down_proj(h)
        h = self.dropout(h)
        return h


class QuantilePredictHead(nn.Module):
    def __init__(self, config: UTP2Config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.mlp_hidden_dim = config.intermediate_size
        self.num_quantiles = len(config.quantiles)
        self.gate_proj = nn.Linear(self.embed_dim, self.mlp_hidden_dim)
        self.up_proj = nn.Linear(self.embed_dim, self.mlp_hidden_dim)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, self.num_quantiles * config.patch_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            h: a tensor of shape (batch_size, seq_len, hidden_size)
        Output:
            (batch_size, seq_len, patch_size * num_quantiles)
        """
        h = torch.sigmoid(self.gate_proj(h)) * self.up_proj(h)
        return self.down_proj(h)


# Modified from https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py
class InstanceNorm(nn.Module):
    """
    Apply standardization along the last dimension and optionally apply arcsinh after standardization.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, x: torch.Tensor, loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale

        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale

        x = x * scale + loc

        return x.to(orig_dtype)


# Copied from https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py
class Patch(nn.Module):
    """
    Patch along the last dimension of the input tensor.
    """
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a tensor of shape (batch_size, seq_len)
        Output:
            (batch_size, num_patches, patch_size)
        """
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device)
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        return x