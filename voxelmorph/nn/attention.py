import torch
import torch.nn as nn

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels, heads=4, dim_head=32):
        super().__init__()
        hidden_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Q, K, V projections
        self.to_qkv = nn.Conv3d(in_channels, hidden_dim * 3, kernel_size=1, bias=False)
        # Output projection
        self.to_out = nn.Conv3d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape

        # Create Q, K, V
        qkv = self.to_qkv(x)  # (B, 3*hidden_dim, D, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head
        def reshape_heads(t):
            return t.view(b, self.heads, -1, d * h * w)  # (B, heads, dim_head, N)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Attention scores
        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # (B, heads, N, N)
        attn_probs = attn_scores.softmax(dim=-1)

        # Weighted sum of values
        out = torch.einsum('bhij,bhjd->bhid', attn_probs, v)  # (B, heads, dim_head, N)

        # Merge heads
        out = out.contiguous().view(b, -1, d, h, w)  # (B, hidden_dim, D, H, W)

        return self.to_out(out)  # project back to original channels












# import torch
# import torch.nn as nn

# class SpatialAttention3D(nn.Module):
#     """
#     Simple spatial attention for 3D feature maps.
#     Input: (B, C, D, H, W)
#     Output: same shape, but features reweighted by a spatial attention map (single channel).
#     """
#
#     def __init__(self, in_channels: int, inter_channels: int = None):
#         super().__init__()
#         if inter_channels is None:
#             inter_channels = max(1, in_channels // 2)
#         # 1x1x1 conv to reduce channels -> nonlinearity -> 1x1x1 conv to single attention map
#         self.attn = nn.Sequential(
#             nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, C, D, H, W)
#         map_ = self.attn(x)    # (B, 1, D, H, W)
#         return x * map_       # broadcast multiply
