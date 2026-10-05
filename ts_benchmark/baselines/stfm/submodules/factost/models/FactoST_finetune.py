__all__ = ['FactoST']

import torch
from torch import nn
from .layers.utp import UTP2, UTP2Config, mask_rope_cos_sin
import math

class FactoST(nn.Module):
    """
    FactoST Fine-Tune Model
    
    Output dimension:
        - [bs x target_dim x nvars] for prediction
        - [bs x target_dim] for regression
        - [bs x target_dim] for classification
    """
    
    def __init__(self, target_dim: int, patch_len: int, n_layers: int = 3, d_model: int = 128, 
                 n_heads: int = 16, d_ff: int = 256, num_token: int = 3, attn_dropout: float = 0., **kwargs):
        
        super().__init__()
        
        # Initialize UTP2 Backbone
        
        config = UTP2Config(
            quantiles=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
            patch_size=patch_len,
            rope_percentage=0.75, 
            max_input_patches=10000,
            max_output_patches=10000,
            hidden_size=d_model,
            intermediate_size=d_ff,
            num_layers=n_layers,
            num_attention_heads=n_heads,
            dropout=attn_dropout,
        )
        
        self.backbone = UTP2(config)
        
        # Low-rank domain adaptation parameters
        self.u = nn.Parameter(torch.zeros(num_token, 1))  # Low-rank matrix U
        self.v = nn.Parameter(torch.rand(1, d_model))     # Low-rank matrix V
        
        # Domain adapter for fine-tuning
        self.domain_adapter = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.domain_adapter.weight)
        if self.domain_adapter.bias is not None:
            nn.init.zeros_(self.domain_adapter.bias)
        
        # Model parameters
        self.d_model = d_model
        self.num_token = num_token
        self.target_dim = target_dim
        self.patch_len = patch_len

    def load_backbone_weights(self, path: str):
        """Load pretrained weights into the backbone."""
        print(f"Loading backbone weights from {path}")
        try:
            # Use UTP2's class method logic but applied to our instance
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Filter for backbone
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Backbone weights loaded successfully.")
            print("Using a very large max_input_patches and max_output_patches for fine-tuning.")
        except Exception as e:
            print(f"Error loading backbone weights: {e}")

    def forward(self, x, mask=None):
        """
        Forward pass for fine-tuning aligned with UTP2 pipeline
        
        Args:
            x: Input tensor [bs, seq_len, n_vars]
            mask: Optional mask
            
        Returns:
            output: Model predictions
        """
        bs, seq_len, n_vars = x.shape
        
        # Flatten for Channel Independence: [bs * n_vars, seq_len]
        x_flat = x.permute(0, 2, 1).reshape(bs * n_vars, seq_len)
        
        if mask is None:
            mask_flat = (~torch.isnan(x_flat)).float()
        else:
            mask_flat = mask.permute(0, 2, 1).reshape(bs * n_vars, seq_len)
            mask_flat = mask_flat & (~torch.isnan(x_flat))
            
        # 1. Prepare patched context (Norm -> Patch -> Mask -> Concat)
        # patched_context: [bs*n_vars, num_input_patches, patch_size*2]
        # attention_mask: [bs*n_vars, num_input_patches]
        # loc_scale: tuple
        patched_context, attention_mask, loc_scale = self.backbone._prepare_patched_context(x_flat, mask_flat)
        
        # 2. Prepare patched future
        num_output_patches = math.ceil(self.target_dim / self.patch_len)
        patched_future = self.backbone._prepare_patched_future(bs * n_vars, num_output_patches)
        
        # 3. Concatenate Context and Future
        patched_context_future = torch.cat([patched_context, patched_future], dim=1)
        
        # 4. Encode patches
        h = self.backbone.ts_encoder(patched_context_future) # [bs*n_vars, num_total_patches, d_model]
        
        # 5. Insert [REG] token (UTP2 logic)
        num_input_patches = patched_context.shape[1]
        reg_token = self.backbone.reg_token.expand(bs * n_vars, -1, -1)
        h = torch.cat([h[:, :num_input_patches], reg_token, h[:, num_input_patches:]], dim=1)
        
        # 6. Domain Adaptation (Prompt Injection)
        # We prepend the prompt to the sequence: [Prompt, Context, REG, Future]
        if self.num_token != 0:
            # Generate low-rank prompt: [num_token, d_model]
            prompt = torch.mm(self.u, self.v)
            # Expand to batch: [bs*n_vars, num_token, d_model]
            prompt = prompt.expand(bs * n_vars, -1, -1)
            
            # Apply domain adaptation
            adapted_prompt = self.domain_adapter(prompt)
            
            # Prepend prompt to hidden states
            h = torch.cat((adapted_prompt, h), dim=1) 
            
            # Extend attention mask for prompt
            prompt_mask = torch.ones(bs * n_vars, self.num_token, dtype=torch.bool, device=h.device)
        else:
            prompt_mask = torch.tensor([], dtype=torch.bool, device=h.device).reshape(bs * n_vars, 0)

        # 7. Build Attention Mask
        # [Prompt(1), Context(Mask), REG(1), Future(1)]
        attention_mask = torch.cat([
            prompt_mask,
            attention_mask,
            torch.ones(bs * n_vars, 1, dtype=torch.bool, device=attention_mask.device), # REG
            torch.ones(bs * n_vars, num_output_patches, dtype=torch.bool, device=attention_mask.device) # Future
        ], dim=1)
        
        # Expand mask for Transformer (B, 1, L, L)
        attention_mask_ext = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        attention_mask_ext = attention_mask_ext.unsqueeze(1)

        # 8. Rotary Embeddings
        position_ids = torch.arange(0, h.shape[1], dtype=torch.long, device=h.device)
        position_ids = position_ids.unsqueeze(0).expand(bs * n_vars, -1)
        cos, sin = self.backbone.rotary_emb(h, position_ids)
        cos, sin = mask_rope_cos_sin(cos, sin, self.backbone.config.rope_percentage)
        
        # 9. Transformer Layers
        for layer in self.backbone.encoder_layers:
            h = layer(h, (cos, sin), attention_mask_ext)
            
        # 10. Prediction Head (UTP2 Head)
        # Predict on last num_output_patches
        outputs = self.backbone.predict_head(h[:, -num_output_patches:]) 
        # outputs: [bs*n_vars, num_output_patches, patch_size * num_quantiles]
        
        # Reshape to [bs*n_vars, num_output_patches * patch_size, num_quantiles]
        outputs = outputs.reshape(bs * n_vars, num_output_patches * self.patch_len, -1)
        
        # Slice to target_dim
        outputs = outputs[:, :self.target_dim, :] # [bs*n_vars, target_dim, num_quantiles]
        
        # 11. Denormalize
        loc, scale = loc_scale
        # loc_scale needs to be expanded to match output shape for inverse
        # loc: [bs*n_vars, 1], scale: [bs*n_vars, 1]
        loc_scale_expanded = (loc.unsqueeze(1), scale.unsqueeze(1)) # ([bs*n_vars, 1, 1], ...)
        
        outputs = self.backbone.instance_norm.inverse(outputs, loc_scale_expanded)
        
        # 12. Final Reshape
        # We assume point forecasting (quantile 0.5), so we take the corresponding channel
        try:
            median_idx = self.backbone.config.quantiles.index(0.5)
        except ValueError:
            # Fallback if 0.5 is not present
            median_idx = len(self.backbone.config.quantiles) // 2
            
        output = outputs[..., median_idx] # [bs*n_vars, target_dim]
        
        output = output.reshape(bs, n_vars, self.target_dim).permute(0, 2, 1) # [bs, target_dim, n_vars]
        
        return output
