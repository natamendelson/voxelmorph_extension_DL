from neurite.nn import models
import torch
import os
import numpy as np
from datetime import datetime
class CustomUNet(models.BasicUNet):
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

        if not self.training:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # base_dir = os.path.join("..", "data")
                base_dir = r"..\data"
                # output_dir = os.path.join(base_dir, f"output_{timestamp}")
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