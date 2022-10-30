import torch
from torch_geometric.data import Data

from data.base_data_loader import BaseDataLoader
import numpy as np


class TseDataLoader(BaseDataLoader):
    def __init__(self, raw_dataset, batch_size=1, graph_depth=5):
        graph_dataset = self.create_graph_dataset(raw_dataset, graph_depth, batch_size)
        super(TseDataLoader, self).__init__(graph_dataset=graph_dataset)

    @staticmethod
    def create_graph_dataset(raw_dataset, graph_depth, batch_size):
        graph_dataset = []
        market_isin = 'IRX6XTPI0009'

        isins = list(raw_dataset.keys())
        for isin in isins:
            raw_dataset[isin] = [
                record if record["vol"] > 0 else
                {"open": 1, "high": 1, "low": 1, "close": 1, "vol": 0, "cap": 0, "count": 0} 
                for record in raw_dataset[isin]]

        for idx in range(len(raw_dataset[market_isin]) - graph_depth - 1):
            node_features = []

            for stock in raw_dataset.keys():
                node_features.append([[
                    +1 if raw_dataset[stock][idx + i + 1]["close"] > raw_dataset[stock][idx + i]["close"] else -1,
                    raw_dataset[stock][idx + i]["high"] / raw_dataset[stock][idx + i]["low"],
                    raw_dataset[stock][idx + i]["open"] / raw_dataset[stock][idx + i]["close"],
                    raw_dataset[stock][idx + i]["high"] / raw_dataset[stock][idx + i]["open"],
                    # raw_dataset[idx + i]["low"] / raw_dataset[idx + i]["close"],
                    # raw_dataset[idx + i]["vol"],
                    # raw_dataset[idx + i]["cap"],
                    # raw_dataset[idx + i]["count"],
                ] for i in range(graph_depth)])

            node_features = np.swapaxes(np.array(node_features), 0, 1)
            edge_index_src = []
            edge_index_dest = []
            for i in range(graph_depth):
                for j in range(i, graph_depth):
                    edge_index_src.append(i)
                    edge_index_dest.append(j)
            edge_index = [[edge_index_src, edge_index_dest]]

            label = [1, 0]
            if raw_dataset[market_isin][idx + graph_depth + 1]["close"] > raw_dataset[market_isin][idx + graph_depth]["close"]:
                label = [0, 1]

            graph_dataset.append(Data(x=torch.tensor(node_features, dtype=torch.float),
                                      edge_index=torch.tensor(edge_index, dtype=torch.long),
                                      y=torch.tensor(label)
                                      ))

        return graph_dataset
