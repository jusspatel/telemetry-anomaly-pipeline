# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List, Tuple, Dict
# import numpy as np

# class CausalConv1d(nn.Module):
#     """
#     1D Dilated Causal Convolution.
#     Applies asymmetric left-padding to guarantee zero future data leakage.
#     """
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, use_weight_norm: bool = False):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         # Calculate exact left padding required: (k - 1) * d
#         self.left_padding = (kernel_size - 1) * dilation
        
#         conv = nn.Conv1d(
#             in_channels, 
#             out_channels, 
#             kernel_size=kernel_size, 
#             stride=1, 
#             padding=0,  # We handle padding manually before convolution
#             dilation=dilation
#         )
        
#         # Apply weight normalization if requested
#         self.conv = nn.utils.weight_norm(conv) if use_weight_norm else conv

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape: (Batch, Channels, Time)
#         # Pad only the left (past) side of the temporal dimension
#         padded_x = F.pad(x, (self.left_padding, 0))
#         return self.conv(padded_x)


# class ResidualBlock(nn.Module):
#     """
#     TCN Residual Block with Weight Normalization, ReLU, Spatial Dropout, and Skip Connections.
#     """
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout_rate: float = 0.1):
#         super().__init__()
        
#         # Two causal convolutional layers per residual block with weight norm applied internally
#         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, use_weight_norm=True)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout_rate)
        
#         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, use_weight_norm=True)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout_rate)
        
#         # If input channels != output channels, use a 1x1 conv to align shapes for the skip connection
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.relu_out = nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         res = x if self.downsample is None else self.downsample(x)
        
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.dropout1(out)
        
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.dropout2(out)
        
#         return self.relu_out(out + res)


# class TCNAutoencoder(nn.Module):
#     """
#     Dilated Causal TCN Autoencoder for F1 Telemetry Sensor Fault Localization.
#     Enforces a strict channel bottleneck (5 -> 16 -> 8 -> 2 -> 8 -> 16 -> 5) across Time=20.
#     """
#     def __init__(self, num_channels: int = 5, latent_dim: int = 2, kernel_size: int = 3, dropout_rate: float = 0.1):
#         super().__init__()
        
#         self.num_channels = num_channels
#         self.latent_dim = latent_dim
        
#         # =========================================================================
#         # ENCODER: Compress 5 Channels -> 2 Latent Variables (Dilations: 1, 2, 4)
#         # =========================================================================
#         self.enc_block1 = ResidualBlock(num_channels, 16, kernel_size, dilation=1, dropout_rate=dropout_rate)
#         self.enc_block2 = ResidualBlock(16, 8, kernel_size, dilation=2, dropout_rate=dropout_rate)
#         self.enc_bottleneck = ResidualBlock(8, latent_dim, kernel_size, dilation=4, dropout_rate=dropout_rate)
        
#         # =========================================================================
#         # DECODER: Expand 2 Latent Variables -> 5 Reconstructed Channels (Dilations: 4, 2, 1)
#         # =========================================================================
#         self.dec_bottleneck = ResidualBlock(latent_dim, 8, kernel_size, dilation=4, dropout_rate=dropout_rate)
#         self.dec_block1 = ResidualBlock(8, 16, kernel_size, dilation=2, dropout_rate=dropout_rate)
#         # Final projection layer outputs raw linear sensor reconstructions (no ReLU at output)
#         self.dec_out = CausalConv1d(16, num_channels, kernel_size=1, dilation=1)

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Passes input through encoder bottleneck and reconstructs.
#         Returns: (reconstructed_tensor, latent_representation)
#         """
#         # Encode
#         e1 = self.enc_block1(x)
#         e2 = self.enc_block2(e1)
#         latent = self.enc_bottleneck(e2)  # Shape: (Batch, Channels=2, Time=20)
        
#         # Decode
#         d1 = self.dec_bottleneck(latent)
#         d2 = self.dec_block1(d1)
#         reconstructed = self.dec_out(d2)  # Shape: (Batch, Channels=5, Time=20)
        
#         return reconstructed, latent

#     def localize_fault(self, raw_window: torch.Tensor, reconstructed_window: torch.Tensor, channel_names: List[str]) -> Dict[str, float]:
#         """
#         Calculates per-channel Mean Absolute Error (MAE) residuals for an anomalous window.
#         Returns the channel exhibiting the highest residual spike as the diagnosed culprit.
#         """
#         # Calculate absolute error matrix: |X - X_hat| -> Shape: (Channels, Time)
#         abs_errors = torch.abs(raw_window - reconstructed_window)
        
#         # Average across the 20 timestamps to get a single MAE score per channel
#         per_channel_mae = torch.mean(abs_errors, dim=-1).squeeze().cpu().detach().numpy()
        
#         results = {}
#         for idx, name in enumerate(channel_names):
#             results[name] = float(per_channel_mae[idx])
            
#         # Identify sensor with the maximum reconstruction residual
#         culprit_channel = max(results, key=results.get)
#         results['DIAGNOSED_CULPRIT'] = culprit_channel
        
#         return results

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
  """1D Dilated Causal Convolution with manual asymmetric left-padding."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      dilation: int = 1,
      use_weight_norm: bool = False,
  ):
    super().__init__()
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.left_padding = (kernel_size - 1) * dilation

    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        dilation=dilation,
    )
    self.conv = nn.utils.weight_norm(conv) if use_weight_norm else conv

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    padded_x = F.pad(x, (self.left_padding, 0))
    return self.conv(padded_x)


class ResidualBlock(nn.Module):
  """TCN Residual Block with GELU activations and an unhindered linear skip connection!"""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      dilation: int,
      dropout_rate: float = 0.1,
  ):
    super().__init__()

    self.conv1 = CausalConv1d(
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        use_weight_norm=True,
    )
    self.act1 = nn.GELU()  # <-- UPGRADED: Allows smooth negative Z-score flow!
    self.dropout1 = nn.Dropout(dropout_rate)

    self.conv2 = CausalConv1d(
        out_channels,
        out_channels,
        kernel_size,
        dilation,
        use_weight_norm=True,
    )
    self.act2 = nn.GELU()
    self.dropout2 = nn.Dropout(dropout_rate)

    self.downsample = (
        nn.Conv1d(in_channels, out_channels, 1)
        if in_channels != out_channels
        else None
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    res = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.act1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.act2(out)
    out = self.dropout2(out)

    # <-- UPGRADED: Removed relu_out! Pure linear addition preserves negative Z-scores!
    return out + res


class TCNAutoencoder(nn.Module):
  """Dilated Causal TCN Autoencoder with widened capacity (5 -> 32 -> 16 -> 4)."""

  def __init__(
      self,
      num_channels: int = 5,
      latent_dim: int = 4,  # <-- UPGRADED: Was 2 (Decouples drivetrain cross-talk)
      kernel_size: int = 3,
      dropout_rate: float = 0.1,
  ):
    super().__init__()

    self.num_channels = num_channels
    self.latent_dim = latent_dim

    # =========================================================================
    # ENCODER: Widen filters to 32 -> 16 -> 4
    # =========================================================================
    self.enc_block1 = ResidualBlock(
        num_channels, 32, kernel_size, dilation=1, dropout_rate=dropout_rate
    )
    self.enc_block2 = ResidualBlock(
        32, 16, kernel_size, dilation=2, dropout_rate=dropout_rate
    )
    self.enc_bottleneck = ResidualBlock(
        16, latent_dim, kernel_size, dilation=4, dropout_rate=dropout_rate
    )

    # =========================================================================
    # DECODER: Mirror encoder expansion 4 -> 16 -> 32 -> 5
    # =========================================================================
    self.dec_bottleneck = ResidualBlock(
        latent_dim,
        16,
        kernel_size,
        dilation=4,
        dropout_rate=dropout_rate,
    )
    self.dec_block1 = ResidualBlock(
        16, 32, kernel_size, dilation=2, dropout_rate=dropout_rate
    )
    self.dec_out = CausalConv1d(32, num_channels, kernel_size=1, dilation=1)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    e1 = self.enc_block1(x)
    e2 = self.enc_block2(e1)
    latent = self.enc_bottleneck(e2)

    d1 = self.dec_bottleneck(latent)
    d2 = self.dec_block1(d1)
    reconstructed = self.dec_out(d2)

    return reconstructed, latent

  def localize_fault(
      self,
      raw_window: torch.Tensor,
      reconstructed_window: torch.Tensor,
      channel_names: List[str],
  ) -> Dict[str, float]:
    abs_errors = torch.abs(raw_window - reconstructed_window)
    per_channel_mae = (
        torch.mean(abs_errors, dim=-1).squeeze().cpu().detach().numpy()
    )

    results = {}
    for idx, name in enumerate(channel_names):
      results[name] = float(per_channel_mae[idx])

    culprit_channel = max(results, key=results.get)
    results["DIAGNOSED_CULPRIT"] = culprit_channel
    return results