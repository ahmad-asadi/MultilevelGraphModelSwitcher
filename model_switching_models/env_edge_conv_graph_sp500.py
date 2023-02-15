import torch
from torch_geometric.loader import DataLoader

from data.stock_markets.sp500_data_loader import SP500DataLoader
from model_switching_models.base_simulation_utils import run
from graph_models.edge_conv_graph_tse import DynamicEdgeConv
from graph_models.market_fusion_graph import DynamicEdgeConv as MarketFusionGraph

from data.stock_markets.sp500_data import load_sp500_stocks_data

data = load_sp500_stocks_data(database_file=None)

train_data_loader = DataLoader(dataset=SP500DataLoader(raw_dataset=data, batch_size=5).graph_dataset, batch_size=1,
                               shuffle=True)

stock_fusion_model = DynamicEdgeConv(in_channels=423, out_channels=16)
market_fusion_model = MarketFusionGraph(in_channels=423, k=3, edge_type="KNN", features=16)

optimizer = torch.optim.Adam(stock_fusion_model.parameters(), lr=0.00005, weight_decay=1e-4)
# criterion = torch.nn.MSELoss(reduction="sum")
criterion = torch.nn.CrossEntropyLoss()

run(stock_fusion_model=stock_fusion_model, market_fusion_model=market_fusion_model,
    train_loader=train_data_loader, criterion=criterion, test_data=train_data_loader,
    optimizer=optimizer, epochs=200)
