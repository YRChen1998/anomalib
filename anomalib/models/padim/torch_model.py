"""PyTorch model for the PaDiM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor, MultiVariateGaussian
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims
from anomalib.models.padim.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.manifold import LocallyLinearEmbedding as LLE
# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}


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


class PadimModel(nn.Module):
    """Padim Module.

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
        n_features: int | None = None,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)
        self.n_features_original, self.n_patches = _deduce_dims(self.feature_extractor, input_size, self.layers)

        n_features = n_features or _N_FEATURES_DEFAULTS.get(self.backbone)

        if n_features is None:
            raise ValueError(
                f"n_features must be specified for backbone {self.backbone}. "
                f"Default values are available for: {sorted(_N_FEATURES_DEFAULTS.keys())}"
            )

        assert (
            0 < n_features <= self.n_features_original
        ), f"for backbone {self.backbone}, 0 < n_features <= {self.n_features_original}, found {n_features}"

        self.n_features = n_features

        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        # self.register_buffer(
        #     "idx",
        #     torch.tensor(sample(range(0, self.n_features_original), self.n_features)),
        # )
        # self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

        # self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
        # Dimension reduction
        # self.PCA = PCA(n_components=n_features)
        # self.LLE = LLE(n_components=n_features)
        # self.PCA_V = PCA()
        self.DFS = PCA()

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            device = embeddings.device
            batch, channel, height, width = embeddings.size()

            # # PCA
            # n_components = self.n_features
            # embeddings = self.PCA.transform(embeddings.permute(0, 2, 3, 1).reshape(-1, channel).cpu().numpy())
            # embeddings = torch.tensor(embeddings).reshape(batch, height, width, n_components).permute(0, 3, 1, 2).float().to(device)

            # LLE
            # n_components = self.n_features
            # embeddings = self.LLE.transform(embeddings.permute(0, 2, 3, 1).reshape(-1, channel).cpu().numpy())
            # embeddings = torch.tensor(embeddings).reshape(batch, height, width, n_components).permute(0, 3, 1, 2).float().to(device)

            # # PCA_V
            # embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, channel).cpu().numpy()
            # embeddings = embeddings - self.PCA_V.mean_
            # if self.Type == 'pca':
            #     embeddings = np.dot(embeddings, self.PCA_V.components_[:self.last_component + 1].T)
            #     embeddings = torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2).to(device)
            # elif self.Type == 'npca':
            #     embeddings = np.dot(embeddings, self.PCA_V.components_[self.last_component - 1:].T)
            #     embeddings = torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2).to(device)

            # DFS
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, channel).cpu().numpy()
            embeddings = embeddings - self.PCA_V.mean_
            if self.Type == '2_3':
                embeddings = np.dot(embeddings, self.DFS.components_[self.m1:].T)
                embeddings = torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2).to(device)
            elif self.Type == '3':
                embeddings = np.dot(embeddings, self.DFS.components_[self.m2:].T)
                embeddings = torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1,2).to(device)
            elif self.Type == '2':
                embeddings = np.dot(embeddings, self.DFS.components_[self.m1:self.m2].T)
                embeddings = torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1,2).to(device)
            elif self.Type == '1':
                embeddings = np.dot(embeddings, self.DFS.components_[:self.m1].T)
                embeddings = torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1,2).to(device)
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, inv_covariance=self.gaussian.inv_covariance
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

        # # subsample embeddings
        # idx = self.idx.to(embeddings.device)
        # embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings

    def pca(self, embeddings: Tensor):
        batch, channel, height, width = embeddings.size()
        n_components = self.n_features

        embeddings = self.PCA.fit_transform(embeddings.permute(0, 2, 3, 1).reshape(-1, channel).numpy())
        embeddings = torch.tensor(embeddings).reshape(batch, height, width, n_components).permute(0, 3, 1, 2)

        return embeddings

    def lle(self, embeddings: Tensor):
        batch, channel, height, width = embeddings.size()
        n_components = self.n_features

        embeddings = self.LLE.fit_transform(embeddings.permute(0, 2, 3, 1).reshape(-1, channel).numpy())
        embeddings = torch.tensor(embeddings).reshape(batch, height, width, n_components).permute(0, 3, 1, 2)

        return embeddings

    def pca_v(self, Type: str, embeddings: Tensor, variance_threshold: float = 0.95):
        batch, channel, height, width = embeddings.size()
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, channel).numpy()
        self.PCA_V.fit(embeddings)
        variances = self.PCA_V.explained_variance_ratio_.cumsum()
        last_component = (variances > variance_threshold).argmax()
        self.last_component = last_component

        embeddings = embeddings - self.PCA_V.mean_
        if Type == 'pca':
            self.Type = Type
            self.n_features = last_component + 1
            embeddings = np.dot(embeddings, self.PCA_V.components_[:last_component + 1].T)
            self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
            return torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2)
        elif Type == 'npca':
            self.Type = Type
            self.n_features = channel - last_component + 1
            embeddings = np.dot(embeddings, self.PCA_V.components_[last_component - 1:].T)
            self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
            return torch.Tensor(embeddings).reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        else:
            raise ValueError("either pca or npca should be specified")

    def dfs(self, Type: str, embeddings: Tensor):
        batch, channel, height, width = embeddings.size()
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, channel).numpy()
        self.DFS.fit(embeddings)
        variance_pca = self.DFS.explained_variance_  # feature矩阵的特征值
        [Ns, Nf] = embeddings.shape
        rw = max(np.nonzero(variance_pca)[0])  # rw:feature矩阵实际的的秩 - 1，即最小非零值的下标
        mr_of_feature = Ns if Ns <= Nf else Nf  # mr_of_feature:feature有可能存在的最大秩
        lmd_med = np.median(variance_pca[:rw + 1])  # 这里rw+1是因为python是左闭右开的，而rw就是下标，所以要+1
        miu = 1
        above_zero = np.maximum(variance_pca - (lmd_med + miu * (lmd_med - variance_pca[rw])), 0)
        m1 = max(np.nonzero(above_zero)[0])
        rk = np.zeros(rw)
        for i in range(rw):
            rk[i] = variance_pca[i + 1] / variance_pca[i]
        window_size = 10
        rk = np.convolve(rk, np.ones(window_size) / window_size, mode="same")  # 滑动平均
        m2 = int(np.where(rk == max(rk[m1 + 1:]))[0][0])  # m2直接就是下标
        self.m1 = m1
        self.m2 = m2
        embeddings = embeddings - self.PCA_V.mean_
        if Type == '2_3':
            self.Type = '2_3'
            self.n_features = channel - m1
            embeddings = np.dot(embeddings, self.DFS.components_[m1:].T)
            self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
            return torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2)
        elif Type == '3':
            self.Type = '3'
            self.n_features = channel - m2
            embeddings = np.dot(embeddings, self.DFS.components_[m2:].T)
            self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
            return torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2)
        elif Type == '2':
            self.Type = '2'
            self.n_features = m2 - m1
            embeddings = np.dot(embeddings, self.DFS.components_[m1:m2].T)
            self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
            return torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2)
        elif Type == '1':
            self.Type = '1'
            self.n_features = m1
            embeddings = np.dot(embeddings, self.DFS.components_[:m1].T)
            self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)
            return torch.Tensor(embeddings).reshape(batch, height, width, self.n_features).permute(0, 3, 1, 2)