import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class CLS_CIFAR10(Dataset):

    def __init__(self, root, cls_dir, csv_file, transform=None):
        self.root = root
        self.cls_dir = cls_dir
        self.cls_files = [f for f in os.listdir(cls_dir) if f.endswith('.pt')]
        self.cls_files = sorted(self.cls_files)
    
        self.data_frame = pd.read_csv(csv_file)
#         self.data = torch.FloatTensor(self.data_frame.values.tolist())
        self.data = torch.LongTensor(self.data_frame.values.tolist())
        self.data = self.data.squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load each load_batch_size items
        load_batch_size = 50
#         load_batch_size = 100

        cls_idx = index//load_batch_size
        cls_vec_name = os.path.join(self.cls_dir, self.cls_files[cls_idx])
        cls_vec = torch.load(cls_vec_name)
        
        cls = cls_vec[index%load_batch_size,:].unsqueeze(0)
#         cls.requires_grad_(False)
        cls = cls.detach()
        cls = cls.cpu()

        label = self.data[index]
#         label.requires_grad_(False)
        label = label.detach()
        label = label.cpu()
        if self.transform:
            cls = self.transform(cls)
        return (cls, label)

def train_dataset(mask_ratio, dataset):
    ROOT = '/data/ek58_data/'
    train_cls_dir = ROOT+'cls_tokens_/'+dataset+'/train_'+str(mask_ratio)+'/data'
    train_csv_file = ROOT+'cls_tokens_/'+dataset+'/train_'+str(mask_ratio)+'/train_cls_y.csv'
    
    train_data = CLS_CIFAR10(root=ROOT, cls_dir=train_cls_dir, csv_file=train_csv_file)
    return train_data

def test_dataset(mask_ratio, dataset):
    ROOT = '/data/ek58_data/'
    test_cls_dir = ROOT+'cls_tokens_/'+dataset+'/test_'+str(mask_ratio)+'/data'
    test_csv_file = ROOT+'cls_tokens_/'+dataset+'/test_'+str(mask_ratio)+'/test_cls_y.csv'

    test_data = CLS_CIFAR10(root=ROOT, cls_dir=test_cls_dir, csv_file=test_csv_file)
    return test_data