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
        self.num_classes = 6  # Citeseer数据集有6个类别
        self.size_bottom_out = 6  # 底部模型输出的大小为类别数

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        # 加载Citeseer数据集
        dataset = Planetoid(root=file_path, name='Citeseer')

        # 获取图数据
        data = dataset[0]

        # 获取训练集、验证集、测试集的索引
        train_labeled_idxs = data.train_mask.nonzero(as_tuple=True)[0].tolist()
        train_unlabeled_idxs = [idx for idx in range(data.num_nodes) if idx not in train_labeled_idxs]

        # 进行标记数据和未标记数据的分割
        # 注意：在Citeseer数据集上，train_mask已经标识了哪些数据是用于训练的
        train_labeled_dataset = self.create_labeled_dataset(data, train_labeled_idxs)
        train_unlabeled_dataset = self.create_unlabeled_dataset(data, train_unlabeled_idxs)

        # 完整训练集
        train_complete_dataset = self.create_labeled_dataset(data, None)

        # 测试集
        test_dataset = self.create_labeled_dataset(data, data.test_mask.nonzero(as_tuple=True)[0].tolist())

        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def create_labeled_dataset(self, data, labeled_idxs):
        # 根据已标记的索引构造数据集
        if labeled_idxs is None:
            labeled_idxs = range(data.num_nodes)
        x = data.x[labeled_idxs]  # 节点特征
        y = data.y[labeled_idxs]  # 节点标签
        return torch.utils.data.TensorDataset(x, y)

    def create_unlabeled_dataset(self, data, unlabeled_idxs):
        # 根据未标记的索引构造数据集
        x = data.x[unlabeled_idxs]  # 节点特征
        return torch.utils.data.TensorDataset(x)

    def get_transforms(self):
        # Citeseer数据集的特征已经准备好了，所以不需要额外的transform
        return None

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        # Citeseer数据集不需要进行转换，直接加载
        dataset = Planetoid(root=file_path, name='Citeseer')
        return dataset

    def clip_one_party_data(self, x, half):
        # 对于Citeseer数据集，通常没有图像裁剪的需求
        # 如果需要分配数据到不同party，可以在这里加入数据划分逻辑
        return x  # 暂时不需要做裁剪

