from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms

gzsl_to_zsl_label_indexes = {3: 0, 10: 1, 23: 2, 24: 3, 32: 4, 38: 5, 53: 6, 57: 7, 72: 8, 74: 9, 75: 10, 85: 11, 95: 12,
99: 13, 103: 14, 112: 15, 124: 16, 130: 17, 138: 18, 145: 19, 152: 20, 158: 21, 184: 22, 196: 23, 216: 24, 221: 25, 237: 26,
245: 27, 246: 28, 254: 29, 259: 30, 262: 31, 286: 32, 298: 33, 315: 34, 328: 35, 336: 36, 342: 37, 353: 38, 358: 39, 379:
40, 381: 41, 420: 42, 423: 43, 425: 44, 440: 45, 448: 46, 471: 47, 482: 48, 493: 49, 508: 50, 509: 51, 517: 52, 529: 53,
558: 54, 560: 55, 580: 56, 622: 57, 631: 58, 635: 59, 645: 60, 650: 61, 656: 62, 658: 63, 674: 64, 679: 65, 681: 66, 695: 67,
710: 68, 711: 69, 712: 70, 715: 71}

class SunDataset(Dataset):
    def __init__(self, indexes, files, labels , data_root, zsl = False,  transform=None, corruption_method=None, corruption_severity=None):
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
        label = torch.Tensor(label.astype(int))

        return (img_tensor, label)

    def __getitem__(self, idx):
        batch = self.fetch_batch(idx)

        return batch
