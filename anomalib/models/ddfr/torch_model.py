"""PyTorch model for the DRAEM model implementation."""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from anomalib.models.components import FeatureExtractor
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims
from anomalib.models.ddfr.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler


def _deduce_dims(
    feature_extractor: FeatureExtractor, input_size: tuple[int, int], layers: list[str]
) -> tuple[int, int]:
    """Run a dry run to deduce the dimensions of the extracted features.

    Important: `layers` is assumed to be ordered and the first (layers[0])
                is assumed to be the layer with largest resolution.

    Returns:
        tuple[int, int]: Dimensions of the extracted features: (n_dims_original, n_patches)
    """
    dimensions_mapping = dryrun_find_featuremap_dims(feature_extractor, input_size, layers)

    # the first layer in `layers` has the largest resolution
    first_layer_resolution = dimensions_mapping[layers[0]]["resolution"]
    n_patches = torch.tensor(first_layer_resolution).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dimensions_mapping[layer]["num_features"] for layer in layers)  # type: ignore

    return n_features_original, n_patches


class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # if 1x1 conv to reconstruct the rgb values, we try to learn a linear combination
        # of the features for rgb
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]
        # layers += [nn.ReLU()]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DdfrModel(nn.Module):
    """
    DDFR Pytorch model consisting of the pre-trained network and reconstructive network
    Args:
        input_size (tuple[int, int]): Input size for the model.
        layers (list[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        n_features (int, optional): Number of features to retain in the dimension reduction step.
                                Default values from the paper are available for: resnet18 (100), wide_resnet50_2 (550).
    """

    def __init__(
            self,
            input_size: tuple[int, int],
            layers: list[str],
            backbone: str = "resnet18",
            pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)
        self.n_features_original, self.n_patches = _deduce_dims(self.feature_extractor, input_size, self.layers)
        self.reconstructive_network = FeatCAE(in_channels=self.n_features_original, latent_dim=64, is_bn=True)
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

    def forward(self, augmented_tensor: Tensor, original_tensor: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        if self.training:
            if self.tiler:
                augmented_tensor = self.tiler.tile(augmented_tensor)
                original_tensor = self.tiler.tile(original_tensor)

            with torch.no_grad():
                augmented_features = self.feature_extractor(augmented_tensor)
                augmented_embeddings = self.generate_embedding(augmented_features)
                original_features = self.feature_extractor(original_tensor)
                original_embeddings = self.generate_embedding(original_features)

            if self.tiler:
                augmented_embeddings = self.tiler.untile(augmented_embeddings)
                original_embeddings = self.tiler.untile(original_embeddings)

            reconstructive_embeddings = self.reconstructive_network(augmented_embeddings)
            output = reconstructive_embeddings, original_embeddings
        else:
            if self.tiler:
                augmented_tensor = self.tiler.tile(augmented_tensor)

            with torch.no_grad():
                augmented_features = self.feature_extractor(augmented_tensor)
                augmented_embeddings = self.generate_embedding(augmented_features)

            if self.tiler:
                augmented_embeddings = self.tiler.untile(augmented_embeddings)
            reconstructive_embeddings = self.reconstructive_network(augmented_embeddings)
            output = self.anomaly_map_generator(
                reconstructive_embeddings=reconstructive_embeddings, original_embeddings=augmented_embeddings
            )
        return output


    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

