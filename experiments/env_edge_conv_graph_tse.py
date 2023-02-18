import torch
from torch_geometric.loader import DataLoader

from data.tse.tse_data_loader import TseDataLoader
from experiments.base_simulation_utils import run

from data.tse.tse_data import load_tse_indices_data
from ms_models.full_fusion_price_predictor import FullFusionPricePredictor

data = load_tse_indices_data(database_file=None)

train_data_loader = DataLoader(dataset=TseDataLoader(raw_dataset=data, batch_size=5).graph_dataset, batch_size=1,
                               shuffle=True)

# model = DynamicEdgeConv(in_channels=4, out_channels=2, k=6)
model = FullFusionPricePredictor(in_channels=41, out_channels=32, k=6)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# criterion = torch.nn.MSELoss(reduction="sum")
criterion = torch.nn.CrossEntropyLoss()

run(model=model, train_loader=train_data_loader, criterion=criterion, test_data=train_data_loader,
    optimizer=optimizer, epochs=200)
