"""PyTorch model for the PEIF model implementation."""

from __future__ import annotations

from random import sample
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims
from anomalib.models.peif.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

from pyod.models.iforest import IForest
from sklearn.decomposition import PCA

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


class PeifModel(nn.Module):
    """PEIF Module.

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

        self.iforest = IForest(n_estimators=100,
                 max_samples='auto',
                 contamination=1e-8,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 behaviour='old',
                 random_state=None,
                 verbose=0)

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
            embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, channel).cpu().numpy()

            # # PCA
            # embeddings = self.PCA.transform(embeddings)

            # DFS
            embeddings = embeddings - self.DFS.mean_
            if self.Type == '2_3':
                embeddings = np.dot(embeddings, self.DFS.components_[self.m1:].T)
            elif self.Type == '3':
                embeddings = np.dot(embeddings, self.DFS.components_[self.m2:].T)
            elif self.Type == '2':
                embeddings = np.dot(embeddings, self.DFS.components_[self.m1:self.m2].T)
            elif self.Type == '1':
                embeddings = np.dot(embeddings, self.DFS.components_[:self.m1].T)

            scores = torch.tensor(self.iforest.decision_function(embeddings)).reshape(batch, 1, height, width).to(device).float()
            output = self.anomaly_map_generator(scores=scores)
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

        # subsample embeddings
        # idx = self.idx.to(embeddings.device)
        # embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings

    def pca(self, embeddings: Tensor):
        batch, channel, height, width = embeddings.size()
        embeddings = self.PCA.fit_transform(embeddings.permute(0, 2, 3, 1).reshape(-1, channel).numpy())

        return embeddings

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
        embeddings = embeddings - self.DFS.mean_
        if Type == '2_3':
            self.Type = '2_3'
            self.n_features = channel - m1
            embeddings = np.dot(embeddings, self.DFS.components_[m1:].T)
            return embeddings
        elif Type == '3':
            self.Type = '3'
            self.n_features = channel - m2
            embeddings = np.dot(embeddings, self.DFS.components_[m2:].T)
            return embeddings
        elif Type == '2':
            self.Type = '2'
            self.n_features = m2 - m1
            embeddings = np.dot(embeddings, self.DFS.components_[m1:m2].T)
            return embeddings
        elif Type == '1':
            self.Type = '1'
            self.n_features = m1
            embeddings = np.dot(embeddings, self.DFS.components_[:m1].T)
            return embeddings
