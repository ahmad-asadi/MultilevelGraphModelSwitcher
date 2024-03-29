from abc import ABC

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn_graph


# noinspection PyMethodOverriding
class EdgeConv(MessagePassing, ABC):
    def __init__(self, alpha, node_features_fusion_type):
        super().__init__(aggr='mean')  # "Max" aggregation.

        self.node_features_fusion_type = node_features_fusion_type

        self.alpha = alpha


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        # if self.node_features_fusion_type == "CAT_MLP":
        #     tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        #     output = self.mlp(tmp)
        # else:
        #     raise ValueError("invalid node features fusion type.")

        output = x_i + self.alpha * (x_j - x_i)

        return output


# noinspection PyAbstractClass
class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, features, k=6, edge_type="KNN", node_feature_fusion_type="CAT_MLP"):
        super().__init__(alpha=0.1, node_features_fusion_type=node_feature_fusion_type)
        self.k = k
        self.edge_type = edge_type
        self.node_feature_fusion_type = node_feature_fusion_type

        self.node_feature_fusion_layer = torch.nn.Conv1d(in_channels, in_channels, features, stride=1)

    def forward(self, X, batch=None):
        edge_index = knn_graph(X, self.k, batch, loop=False, flow=self.flow)

        # X = self.node_feature_fusion_layer(X)
        # X = torch.squeeze(X)

        output = super().forward(X, edge_index)

        return output
