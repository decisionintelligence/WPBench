__all__ = ['FactoST']

import torch
from torch import nn
from .layers.utp import UTP2, UTP2Config

class FactoST(nn.Module):
    """
    FactoST Zero-Shot Model
        
    Output dimension:
        - [bs x target_dim x nvars] for prediction
        - [bs x target_dim] for regression
        - [bs x target_dim] for classification
    """
    
    def __init__(self, c_in: int, target_dim: int, patch_len: int, n_layers: int = 3, d_model: int = 128, 
                 n_heads: int = 16, d_ff: int = 256, attn_dropout: float = 0., **kwargs):

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
        
        # Model parameters
        self.d_model = d_model
        self.target_dim = target_dim
        self.patch_len = patch_len
    
    def load_backbone_weights(self, path: str):
        """Load pretrained weights into the backbone."""
        print(f"Loading backbone weights from {path}")
        try:
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            if 'config' in state_dict:
                # This is for align max_input_patches and max_output_patches with checkpoint
                print(f"Use backbone config from checkpoint for zeroshot.")
                print(f"max_input_patches: {self.backbone.config.max_input_patches} -> {state_dict['config']['max_input_patches']}")
                print(f"max_output_patches: {self.backbone.config.max_output_patches} -> {state_dict['config']['max_output_patches']}")
                self.backbone.config = UTP2Config(**state_dict['config'])
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Backbone weights loaded successfully.")
        except Exception as e:
            print(f"Error loading backbone weights: {e}")

    def forward(self, x, mask=None):
        """
        Forward pass for zero-shot inference aligned with UTP2 pipeline
        
        Args:
            x: Input tensor [bs, seq_len, n_vars]
            mask: Optional mask
            
        Returns:
            output: Model predictions
        """
        bs, seq_len, n_vars = x.shape
        
        # Flatten for Channel Independence: [bs * n_vars, seq_len]
        x_flat = x.permute(0, 2, 1).reshape(bs * n_vars, seq_len)
        
        # Apply mask if provided by setting masked values to NaN
        if mask is not None:
            mask_flat = mask.permute(0, 2, 1).reshape(bs * n_vars, seq_len)
            x_flat = x_flat.clone()
            x_flat[mask_flat == 0] = float('nan')
            
        # Use UTP2's predict method
        # It handles patching, encoding, predicting, and denormalizing internally.
        # Returns: [bs*n_vars, target_dim, num_quantiles]
        outputs = self.backbone.predict(x_flat, prediction_length=self.target_dim)
        
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
