import torch
from torch.nn import Flatten, Linear, Softmax, Sequential as Seq

from graph_models.edge_conv_graph_tse import DynamicEdgeConv
from graph_models.market_fusion_graph import DynamicEdgeConv as MarketFusionGraph


class FullFusionPricePredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(FullFusionPricePredictor, self).__init__()
        self.stock_fusion_model = DynamicEdgeConv(in_channels=in_channels, out_channels=out_channels)
        self.market_fusion_model = MarketFusionGraph(in_channels=in_channels, k=k, edge_type="KNN",
                                                     features=out_channels)
        self.output_generator_model = Seq(
            Flatten(start_dim=0),
            Linear(in_channels * out_channels, int(0.5 * in_channels * out_channels)),
            Linear(int(0.5 * in_channels * out_channels), 2),
            Softmax(dim=-1)
        )

    def forward(self, X, batch=None):
        features = self.stock_fusion_model(X)

        market_features = self.market_fusion_model(features)

        output = self.output_generator_model(market_features)

        return output

