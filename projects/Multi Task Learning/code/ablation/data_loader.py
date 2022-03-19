import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader

import math
import h5py
import numpy as np
from tqdm import tqdm

def load_data(root_path, batch_size=16, num_workers=2):   
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

    img_path = root_path+'/images.h5'
    img_h5 = h5py.File(img_path , 'r', swmr=True)
    img_key = list(img_h5.keys())[0]
    imgs = img_h5[img_key]

    label_type = ['binary', 'bboxes', 'masks']
    label_path = [root_path+'/'+label_type[i]+'.h5' for i in range(len(label_type))]
    label_h5 = [h5py.File(label_path[i] , 'r', swmr=True) for i in range(len(label_path))]
    label_key = [list(label_h5[i].keys())[0] for i in range(len(label_path))]
    labels = [label_h5[i][label_key[i]] for i in range(len(label_type))]

    dataset = MyDataset(imgs, labels, len(label_type), transform)
    dataloader = DataLoader(dataset, batch_size = 16, shuffle=True, num_workers=num_workers)
    return dataset, dataloader

class MyDataset(Dataset):
    def __init__(self, images, labels, num_label, transform):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.imgs = images
        self.labels = labels
        self.num_label = num_label

    def __getitem__(self, index):
        image = self.imgs[index]
        label = [self.labels[i][index] for i in range(self.num_label)]
        return self.transform(np.array(image, dtype=np.uint8)), label

    def __len__(self):
        return int(self.imgs.len())