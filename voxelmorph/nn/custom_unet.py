from neurite.nn import models
import torch
import os
import numpy as np
from datetime import datetime
from voxelmorph.nn.attention import SpatialAttention3D
from voxelmorph.nn.attention import SelfAttention3D

class CustomUNet(models.BasicUNet):
    def __init__(
            self, *args,
            use_bottleneck_attention=False,
            use_skip_attention=False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        nb_features = kwargs.get('nb_features', getattr(self, 'nb_features', None))
        if nb_features is None:
            in_ch_bottleneck = 22
        else:
            in_ch_bottleneck = nb_features[-1]

        # attention
        self.use_bottleneck_attention = use_bottleneck_attention
        self.use_skip_attention = use_skip_attention

        # attention bottleneck
        if self.use_bottleneck_attention:
            self.bottleneck_attention = SelfAttention3D(
                in_channels=in_ch_bottleneck,
                heads=4,
                dim_head=16
            )

        # attention skip connections
        if self.use_skip_attention:
            self.skip_attentions = torch.nn.ModuleList([
                SpatialAttention3D(
                    in_channels=block.out_channels if hasattr(block, 'out_channels') else 22
                ) for block in self.downsampling_conv_blocks
            ])

    def forward(self, feature_tensor: torch.Tensor):
        skip_connections = []

        for i, down_block in enumerate(self.downsampling_conv_blocks):
            if self.residual_connections:
                feature_tensor, residual = down_block(feature_tensor)

                # attention skip connections
                if self.use_skip_attention:
                    residual = self.skip_attentions[i](residual)

                skip_connections.append(residual)
            else:
                feature_tensor = down_block(feature_tensor)

        # bottleneck
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # attention bottleneck
        if self.use_bottleneck_attention:
            feature_tensor = self.bottleneck_attention(feature_tensor)


        if not self.training:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_dir = r"..\data"
                # print("Use attention:", self.use_bottleneck_attention)
                if self.uuse_bottleneck_attention:
                    feature_tensor = self.bottleneck_attention(feature_tensor)
                    output_dir = os.path.join(base_dir, "outputs_attention")
                else:
                    output_dir = os.path.join(base_dir, "outputs")
                os.makedirs(output_dir, exist_ok=True)
                bottleneck_np = feature_tensor.detach().cpu().numpy()
                filename = os.path.join(output_dir, f"bottleneck_features_{timestamp}.npy")
                np.save(filename, bottleneck_np)
                print(f"Bottleneck features saved to {filename}, with shape {bottleneck_np.shape}")
            except:
                print("Bottleneck features could not be saved")

        # Upsampling path
        for i, upsampling_conv_block in enumerate(self.upsampling_conv_blocks):
            if self.residual_connections:
                skip = skip_connections[-(i + 1)]
                feature_tensor = upsampling_conv_block(feature_tensor, skip)
            else:
                feature_tensor = upsampling_conv_block(feature_tensor)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)
        return feature_tensor