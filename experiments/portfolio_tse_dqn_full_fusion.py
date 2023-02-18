import torch
from torch_geometric.loader import DataLoader
from data.tse.tse_data_loader import TseDataLoader
from drl.dqn import DQN
from experiments.base_simulation_utils import run
from graph_models.edge_conv_graph_tse import DynamicEdgeConv
from graph_models.market_fusion_graph import DynamicEdgeConv as MarketFusionGraph
from data.tse.tse_data import load_tse_indices_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_tse_indices_data(database_file=None)

train_data_loader = DataLoader(dataset=TseDataLoader(raw_dataset=data, batch_size=5).graph_dataset, batch_size=1,
                               shuffle=True)

# model = DynamicEdgeConv(in_channels=4, out_channels=2, k=6)
stock_fusion_model = DynamicEdgeConv(in_channels=41, out_channels=32)
market_fusion_model = MarketFusionGraph(in_channels=41, k=6, edge_type="KNN", features=32)

policy_net = DQN(1, 1).to(device)
target_net = DQN(1, 1).to(device)

optimizer = torch.optim.Adam(stock_fusion_model.parameters() + market_fusion_model.parameters() + policy_net.parameters(), lr=0.0001, weight_decay=1e-4)
# criterion = torch.nn.MSELoss(reduction="sum")
criterion = torch.nn.CrossEntropyLoss()

run(stock_fusion_model=stock_fusion_model, market_fusion_model=market_fusion_model,
    train_loader=train_data_loader, criterion=criterion, test_data=train_data_loader,
    optimizer=optimizer, epochs=200)
