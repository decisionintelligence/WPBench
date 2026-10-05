import torch
from torch import nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series data
    
    This module performs instance normalization that can be reversed exactly,
    making it particularly suitable for time series forecasting tasks.
    The normalization is performed along the temporal dimension.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, device=None):
        """
        Initialize RevIN module
        
        Args:
            num_features (int): Number of features/channels in input
            eps (float): Small constant for numerical stability
            affine (bool): If True, applies learnable affine transformation after normalization
            device (torch.device): Device to place the parameters on
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = device
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Forward pass for both normalization and denormalization
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, sequence_length, num_features]
            mode (str): Operation mode - either "norm" for normalization or "denorm" for denormalization
        
        Returns:
            Tensor: Normalized or denormalized tensor, depending on mode
        """
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """
        Initialize learnable affine transformation parameters
        Creates scale (affine_weight) and shift (affine_bias) parameters for each feature
        """
        self.affine_weight = nn.Parameter(torch.ones(self.num_features, device=self.device))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features, device=self.device))

    def _get_statistics(self, x):
        """
        Calculate mean and standard deviation along temporal dimension, channel independent normalization
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, sequence_length, num_features]
            
        Stores:
            mean: Mean along temporal dimension
            stdev: Standard deviation along temporal dimension
        """
        dim2reduce = tuple(range(1, x.ndim - 1))  # Reduce along temporal dimension, x.ndim means the dimension of the tensor (here is 3) => (1,)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        """
        Normalize input tensor using instance statistics
        
        Steps:
        1. Center data by subtracting mean
        2. Scale data by dividing by standard deviation
        3. Apply affine transformation if enabled
        """
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Reverse the normalization process exactly
        
        Steps:
        1. Reverse affine transformation if enabled
        2. Rescale data by multiplying with standard deviation
        3. Recenter data by adding mean
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        self.stdev = self.stdev.to(x.device)
        self.mean = self.mean.to(x.device)
        x = x * self.stdev
        x = x + self.mean
        return x
