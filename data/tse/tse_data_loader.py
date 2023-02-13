import torch
import numpy as np
import talib as ta

from torch_geometric.data import Data
from data.base_data_loader import BaseDataLoader


class TseDataLoader(BaseDataLoader):
    def __init__(self, raw_dataset, batch_size=1, graph_depth=5):
        graph_dataset = self.create_graph_dataset(raw_dataset, graph_depth, batch_size)
        super(TseDataLoader, self).__init__(graph_dataset=graph_dataset)

    def create_graph_dataset(self, raw_dataset, graph_depth, batch_size):
        graph_dataset = []
        market_isin = 'IRX6XTPI0009'

        isins = list(raw_dataset.keys())
        for isin in isins:
            raw_dataset[isin] = [
                record if record["vol"] > 0 else
                {"open": 1, "high": 1, "low": 1, "close": 1, "vol": 0, "cap": 0, "count": 0}
                for record in raw_dataset[isin]]

        for idx in range(40, len(raw_dataset[market_isin]) - graph_depth - 1):  # first idx should be 40 for
            # computing long_MA indicator
            node_features = []

            for stock in raw_dataset.keys():
                node_f = []
                for i in range(graph_depth):
                    candle_representation = self.get_candle_representation(i, raw_dataset=raw_dataset,
                                                                           stock=stock, idx=idx)
                    trend_representation = self.get_trend_representation(i, raw_dataset=raw_dataset,
                                                                         stock=stock, idx=idx)
                    volatility_representation = self.get_volatility_representation(i, raw_dataset=raw_dataset,
                                                                                   stock=stock, idx=idx)

                    node_f.append(candle_representation + trend_representation + volatility_representation)
                node_features.append(node_f)

            node_features = np.swapaxes(np.array(node_features), 0, 1)
            edge_index_src = []
            edge_index_dest = []
            for i in range(graph_depth):
                for j in range(i, graph_depth):
                    edge_index_src.append(i)
                    edge_index_dest.append(j)
            edge_index = [[edge_index_src, edge_index_dest]]

            label = [1, 0]
            if raw_dataset[market_isin][idx + graph_depth + 1]["close"] > raw_dataset[market_isin][idx + graph_depth][
                "close"]:
                label = [0, 1]

            graph_dataset.append(Data(x=torch.tensor(node_features, dtype=torch.float),
                                      edge_index=torch.tensor(edge_index, dtype=torch.long),
                                      y=torch.tensor(label)
                                      ))

        return graph_dataset

    @staticmethod
    def get_candle_representation(i, raw_dataset, stock, idx):
        representation = [
            +1 if raw_dataset[stock][idx + i + 1]["close"] > raw_dataset[stock][idx + i]["close"] else -1,
            raw_dataset[stock][idx + i]["high"] / raw_dataset[stock][idx + i]["low"],
            raw_dataset[stock][idx + i]["open"] / raw_dataset[stock][idx + i]["close"],
            raw_dataset[stock][idx + i]["high"] / raw_dataset[stock][idx + i]["open"],
        ]

        return representation

    @staticmethod
    def get_trend_representation(i, raw_dataset, stock, idx):
        closes = np.array([raw_dataset[stock][idx + i - ma_ind]["close"] for ma_ind in range(40)], dtype=np.double)
        short_ma = ta.EMA(closes, timeperiod=5)
        middle_ma = ta.EMA(closes, timeperiod=10)
        long_ma = ta.EMA(closes, timeperiod=20)
        macd_res = ta.MACD(closes)

        representation = [
            short_ma[-1] / middle_ma[-1] - 1,
            middle_ma[-1] / long_ma[-1] - 1,
            short_ma[-1] / long_ma[-1] - 1,
            macd_res[0][-1],
            macd_res[1][-1],
            macd_res[2][-1],
        ]

        return representation

    @staticmethod
    def get_volatility_representation(i, raw_dataset, stock, idx):
        closes = np.array([raw_dataset[stock][idx + i - ma_ind]["close"] for ma_ind in range(40)], dtype=np.double)

        bb = ta.BBANDS(closes)

        representation = [
            bb[0][-1] / bb[1][-1],
            bb[1][-1] / bb[2][-1],
            bb[0][-1] / bb[2][-1],
        ]

        return representation
