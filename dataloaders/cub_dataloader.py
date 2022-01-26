from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms

gzsl_to_zsl_label_indexes = {6: 0, 18: 1, 20: 2, 28: 3, 33: 4, 35: 5, 49: 6, 55: 7, 61: 8, 67: 9, 68: 10, 71: 11, 78: 12,
79: 13, 86: 14, 87: 15, 90: 16, 94: 17, 97: 18, 99: 19, 103: 20, 107: 21, 115: 22, 119: 23, 121: 24, 123: 25, 124: 26,
128: 27, 138: 28, 140: 29, 141: 30, 149: 31, 151: 32, 156: 33, 158: 34, 159: 35, 165: 36, 166: 37, 170: 38, 173: 39, 175: 40,
178: 41, 181: 42, 184: 43, 186: 44, 188: 45, 190: 46, 191: 47, 192: 48, 194: 49}

class CUBDataset(Dataset):
    def __init__(self, indexes, files, labels , data_root, zsl = False, transform=None, corruption_method=None, corruption_severity=None):
        self.index_instances = indexes
        self.data_root = data_root

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.file_names = files[self.index_instances - 1]
        self.labels = (labels[self.index_instances - 1] -1)

        self.method= corruption_method
        self.severity= corruption_severity
        self.toTensor = transforms.Compose([transforms.ToTensor()])

        if self.method is None:
            self.is_corruption = False
        else:
            self.is_corruption = True
            print("Dataloader:",  self.method.__name__ , "_", self.severity)

        if zsl:
            self.map_labels_zsl()

        self.transform = transform

    def __len__(self):
        return len(self.index_instances)

    def map_labels_zsl(self):
        for index, label in enumerate(self.labels):
            self.labels[index] = gzsl_to_zsl_label_indexes[label[0][0]]

    def fetch_batch(self, idx):
        im_name = self.file_names[idx][0][0][0].split('images/')[1]
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
        img_tensor = self.normalize(img_tensor)

        label = self.labels[idx]
        label = label[0]
        label = torch.Tensor(label)

        return (img_tensor, label)

    def __getitem__(self, idx):
        batch = self.fetch_batch(idx)

        return batch


