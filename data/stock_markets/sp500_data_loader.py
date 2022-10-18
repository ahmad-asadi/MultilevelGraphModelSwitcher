import torch
from torch_geometric.data import Data

from data.base_data_loader import BaseDataLoader


class SP500DataLoader(BaseDataLoader):
    def __init__(self, raw_dataset, batch_size=1, graph_depth=5):
        graph_dataset = self.create_graph_dataset(raw_dataset, graph_depth, batch_size)
        super(SP500DataLoader, self).__init__(graph_dataset=graph_dataset)

    @staticmethod
    def create_graph_dataset(raw_dataset, graph_depth, batch_size):
        graph_dataset = []

        for idx in range(len(raw_dataset) - graph_depth):
            node_features = [[
                    raw_dataset[idx + i]["open"]/raw_dataset[idx + i]["open"],
                    raw_dataset[idx + i]["high"]/raw_dataset[idx + i]["high"],
                    raw_dataset[idx + i]["low"]/raw_dataset[idx + i]["low"],
                    raw_dataset[idx + i]["close"]/raw_dataset[idx + i]["close"],
                    # raw_dataset[idx + i]["vol"],
                    # raw_dataset[idx + i]["cap"],
                    # raw_dataset[idx + i]["count"],
                ] for i in range(graph_depth)]

            # TODO: current edge_index creates a linear connectivity over time. We should examine short-links between
            #  nodes during time. E.g. the connection between node 0 and node 2 or node 0 and node 4 should be
            #  considered.
            edge_index = [[
                list(range(graph_depth-1)),
                list(range(1, graph_depth))
            ] for _ in range(graph_depth)]

            graph_dataset.append(Data(x=torch.tensor(node_features, dtype=torch.float),
                                      edge_index=torch.tensor(edge_index, dtype=torch.long)))

        return graph_dataset
