import torch

from graph_models.edge_conv_graph_tse import DynamicEdgeConv
from graph_models.market_fusion_graph import DynamicEdgeConv as MarketFusionGraph


class FullFusionPricePredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(FullFusionPricePredictor, self).__init__()
        self.stock_fusion_model = DynamicEdgeConv(in_channels=in_channels, out_channels=out_channels)
        self.market_fusion_model = MarketFusionGraph(in_channels=in_channels, k=k, edge_type="KNN",
                                                     features=out_channels)

    def forward(self, X, batch=None):
        features = self.stock_fusion_model(X)

        output = self.market_fusion_model(features)

        return output

