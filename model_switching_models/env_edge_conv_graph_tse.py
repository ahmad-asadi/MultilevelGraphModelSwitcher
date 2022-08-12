import torch

from model_switching_models.base_simulation_utils import run
from graph_models.edge_conv_graph_tse import DynamicEdgeConv

model = DynamicEdgeConv(in_channels=5, out_channels=5, k=6)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
run(model, train_loader=None, criterion=None, test_data=None, epochs=5)

