from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms

class AWADataset(Dataset):
    def __init__(self, indexes, files, labels, data_root, zsl=False, transform=None, corruption_method=None, corruption_severity=None):

        self.index_instances = indexes
        self.data_root = data_root

        self.file_names = files[self.index_instances - 1]
        self.labels = (labels[self.index_instances - 1] -1)
        self.method= corruption_method
        self.severity= corruption_severity
        self.toTensor = transforms.Compose([transforms.ToTensor()])

        if self.method is None:
            self.is_corruption = False
        else:
            self.is_corruption = True
            print("Dataloader:", self.method.__name__, "_", self.severity)
        if zsl:
            self.map_labels_zsl()

        self.transform = transform

    def __len__(self):
        return len(self.index_instances)

    def map_labels_zsl(self):
        for index, i in enumerate(self.labels):
            if i == 6:
                self.labels[index] = 0
            elif i == 8:
                self.labels[index] = 1
            elif i == 22:
                self.labels[index] = 2
            elif i == 23:
                self.labels[index] = 3
            elif i == 29:
                self.labels[index] = 4
            elif i == 30:
                self.labels[index] = 5
            elif i == 33:
                self.labels[index] = 6
            elif i == 40:
                self.labels[index] = 7
            elif i == 46:
                self.labels[index] = 8
            elif i == 49:
                self.labels[index] = 9
            else:
                print("Wrong label for zsl conversion!", i)

    def fetch_batch(self, idx):
        im_name = self.file_names[idx][0][0][0].split('JPEGImages/')[1]

        image_file = os.path.join(self.data_root, im_name)

        img_pil = Image.open(image_file).convert("RGB")
        img_pil = self.transform(img_pil)

        if self.is_corruption:
            try:
                img_pil = self.method(img_pil, self.severity)
            except:
                print("Failure:", self.method.__name__, self.severity, im_name)
                img_pil = np.asarray(img_pil)

            img_pil = Image.fromarray(np.uint8(img_pil))

        img_tensor = self.toTensor(img_pil)
        label = self.labels[idx]
        label = label[0]
        label = torch.Tensor(label)

        return (img_tensor, label)

    def __getitem__(self, idx):
        batch = self.fetch_batch(idx)
        return batch



