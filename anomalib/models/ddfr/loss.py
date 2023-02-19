"""Loss function for the DDFR model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from kornia.losses import FocalLoss, SSIMLoss
from torch import Tensor, nn


class DDFRLoss(nn.Module):
    """Overall loss function of the DDFR model.

    The total loss consists of the sum of the L2 loss and Focal loss between the reconstructed image and the input
    image, and the Structural Similarity loss between the predicted and GT anomaly masks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.l2_loss = nn.modules.loss.MSELoss()

    def forward(self, reconstructive_embeddings: Tensor, original_embeddings: Tensor) -> Tensor:
        """Compute the loss over a batch for the DDFR model."""
        l2_loss_val = self.l2_loss(reconstructive_embeddings, original_embeddings)
        # focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        # ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        return l2_loss_val
