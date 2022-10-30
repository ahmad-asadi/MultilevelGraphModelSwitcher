from abc import ABC

import torch
from torch import Tensor
from torch.nn import Sequential as Seq, Linear, ReLU, Flatten, Tanh, Softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn_graph


# noinspection PyMethodOverriding
class EdgeConv(MessagePassing, ABC):
    def __init__(self, in_channels, out_channels, node_features_fusion_type):
        super().__init__(aggr='mean')  # "Max" aggregation.

        self.node_features_fusion_type = node_features_fusion_type

        self.mlp = Seq(Linear(2 * in_channels, in_channels),
                       # ReLU(),
                       Linear(in_channels, out_channels),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        if self.node_features_fusion_type == "CAT_MLP":
            tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
            output = self.mlp(tmp)
        else:
            raise ValueError("invalid node features fusion type.")

        return output


# noinspection PyAbstractClass
class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6, edge_type="LINEAR", node_feature_fusion_type="CAT_MLP"):
        super().__init__(in_channels, in_channels, node_feature_fusion_type)
        self.k = k
        self.edge_type = edge_type
        self.node_feature_fusion_type = node_feature_fusion_type

        self.output_generator_model = Seq(
            Flatten(0, -1),
            Linear(5 * in_channels, in_channels),
            Linear(in_channels, in_channels),
            Linear(in_channels, out_channels),
            Softmax(dim=-1)
        )

        self.node_features_fusion_layer = torch.nn.Conv1d(41, 41, 4, stride=1)

    def forward(self, X, batch=None):
        if self.edge_type == "KNN":
            x = X
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        elif self.edge_type == "LINEAR":
            x = X["x"]
            edge_index = X["edge_index"][0]
        else:
            raise ValueError("invalid edge type")

        x = self.node_features_fusion_layer(x)
        x = torch.squeeze(x)

        graph_output = super().forward(x, edge_index)

        output = self.output_generator_model(graph_output)

        return output
