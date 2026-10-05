import torch
from .transforms import RevInCB

class RevInCBWrapper(RevInCB):
    def __init__(self, num_features: int, eps=1e-5, affine:bool=False, denorm:bool=True, device=None):
        """
        Wrapper for RevIN callback that only normalizes the first feature (value feature)
        while keeping other features (time features) unchanged.
        """
        self.device = device
        super().__init__(num_features, eps, affine, denorm, device)
    
    def revin_norm(self):
        """
        Normalize only the first feature (value feature) while keeping time features unchanged
        """
        # print("\n=== Before RevIN ===")
        # print(f"Input shape: {self.xb.shape}")  # [samples, nodes, 3] (value, time_of_day, day_of_week)
        
        xb_revin = self.xb.clone()
        xb_revin[..., 0] = self.revin(self.xb[..., 0], 'norm').to(self.device)
        self.learner.xb = xb_revin
        
        # print("\n=== After RevIN ===")
        # print(f"Output shape: {self.learner.xb.shape}") # [batch_size, seq_len, num_nodes, 3] (value, time_of_day, day_of_week) only value is normalized

    def revin_denorm(self):
        """
        Denormalize only the predictions (which only contain value feature)
        """
        # print("\n=== Before RevIN Denorm ===")
        # print(f"Input shape: {self.pred.shape}")    # [batch_size, target_points, num_nodes]
        
        self.learner.pred = self.revin(self.pred, 'denorm').to(self.device)
        
        # print("\n=== After RevIN Denorm ===")
        # print(f"Output shape: {self.learner.pred.shape}") # [batch_size, target_points, num_nodes]
    

