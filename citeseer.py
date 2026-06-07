from torch_geometric.datasets import Planetoid
import torch
import numpy as np
import torchvision
from torchvision import transforms, datasets
from label_inference_attacks.Code.datasets.dataset_setup import DatasetSetup
from label_inference_attacks.Code.my_utils.utils import train_val_split


class CiteseerSetup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 6
        self.size_bottom_out = 6

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        dataset = Planetoid(root=file_path, name='Citeseer')

        data = dataset[0]

        train_labeled_idxs = data.train_mask.nonzero(as_tuple=True)[0].tolist()
        train_unlabeled_idxs = [idx for idx in range(data.num_nodes) if idx not in train_labeled_idxs]

        train_labeled_dataset = self.create_labeled_dataset(data, train_labeled_idxs)
        train_unlabeled_dataset = self.create_unlabeled_dataset(data, train_unlabeled_idxs)

        train_complete_dataset = self.create_labeled_dataset(data, None)

        test_dataset = self.create_labeled_dataset(data, data.test_mask.nonzero(as_tuple=True)[0].tolist())

        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def create_labeled_dataset(self, data, labeled_idxs):
        if labeled_idxs is None:
            labeled_idxs = range(data.num_nodes)
        x = data.x[labeled_idxs]
        y = data.y[labeled_idxs]
        return torch.utils.data.TensorDataset(x, y)

    def create_unlabeled_dataset(self, data, unlabeled_idxs):
        x = data.x[unlabeled_idxs]
        return torch.utils.data.TensorDataset(x)

    def get_transforms(self):
        return None

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        dataset = Planetoid(root=file_path, name='Citeseer')
        return dataset

    def clip_one_party_data(self, x, half):
        return x

