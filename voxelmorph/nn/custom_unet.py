from neurite.nn import models
import torch
import os
import numpy as np
from datetime import datetime
# from voxelmorph.nn.attention import SpatialAttention3D
from voxelmorph.nn.attention import SelfAttention3D

class CustomUNet(models.BasicUNet):
    def __init__(self, *args, use_attention=True, **kwargs):
        super().__init__(*args, **kwargs)


        nb_features = kwargs.get('nb_features', getattr(self, 'nb_features', None))
        if nb_features is None:
            in_ch_bottleneck = 22
        else:
            in_ch_bottleneck = nb_features[-1]

        # attention

        self.use_attention = use_attention

        # if self.use_attention:
        #     self.bottleneck_attention = SpatialAttention3D(
        #         in_channels=in_ch_bottleneck,
        #         inter_channels=max(1, in_ch_bottleneck // 2)
        #     )
        if self.use_attention:
            self.bottleneck_attention = SelfAttention3D(
                in_channels=in_ch_bottleneck,
                heads=4,  # אפשר לשחק עם מספר הראשים
                dim_head=16  # גודל כל ראש
            )



    def forward(self, feature_tensor: torch.Tensor):
        """
        Forward pass through the `BasicUNet` model.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """

        # Downsampling path
        skip_connections = []

        for downsampling_conv_block in self.downsampling_conv_blocks:
            if self.residual_connections:
                feature_tensor, residual = downsampling_conv_block(feature_tensor)
                skip_connections.append(residual)  # Save for skip connection
            else:
                feature_tensor = downsampling_conv_block(feature_tensor)

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)  # bottleneck

        # attention


        if not self.training:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_dir = r"..\data"
                print("Use attention:", self.use_attention)
                if self.use_attention:
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