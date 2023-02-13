from abc import ABC

import torch
from torch import Tensor
from torch.nn import Sequential as Seq, Linear, ReLU, Flatten, Tanh, Softmax, Conv2d
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
            Linear(15, out_channels),
            Softmax(dim=-1)
        )

        self.candle_features_fusion_layer = torch.nn.Conv1d(in_channels, in_channels, 4, stride=1)
        self.trend_features_fusion_layer = torch.nn.Conv1d(in_channels, in_channels, 6, stride=1)
        self.volatility_features_fusion_layer = torch.nn.Conv1d(in_channels, in_channels, 3, stride=1)

    def forward(self, X, batch=None):
        if self.edge_type == "KNN":
            x = X
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        elif self.edge_type == "LINEAR":
            x = X["x"]
            edge_index = X["edge_index"][0]
        else:
            raise ValueError("invalid edge type")

        ohlc_x = x[:, :, :4]
        ohlc_x = self.candle_features_fusion_layer(ohlc_x)
        ohlc_x = torch.squeeze(ohlc_x)
        candle_graph_output = super().forward(ohlc_x, edge_index)

        trend_x = x[:, :, 4:10]
        trend_x = self.trend_features_fusion_layer(trend_x)
        trend_x = torch.squeeze(trend_x)
        trend_graph_output = super().forward(trend_x, edge_index)

        volatility_x = x[:, :, 10:]
        volatility_x = self.volatility_features_fusion_layer(volatility_x)
        volatility_x = torch.squeeze(volatility_x)
        volatility_graph_output = super().forward(volatility_x, edge_index)

        stock_f = torch.swapaxes(torch.concat([candle_graph_output, trend_graph_output, volatility_graph_output]), 0, 1)

        output = self.output_generator_model(stock_f)

        return output
