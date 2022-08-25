from torch.utils.data import Dataset


class BaseDataLoader(Dataset):
    def __init__(self, graph_dataset):
        self.graph_dataset = graph_dataset

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        return self.graph_dataset[idx]
