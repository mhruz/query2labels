import torch
import sys
import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm


class NAKIDataset(data.Dataset):
    def __init__(self, image_dir, anno_path, input_transform=None, read_to_mem=False):

        self.data = {"images": [], "targets": []}
        self.image_dir = image_dir
        self.classes = {"signature": 0, "handwritten": 1, "title": 2, "stamp": 3, "typewritten": 4, "plaintext": 5,
                        "fingerprint": 6, "photo": 7, "other": 8, "list": 9}
        self.classes_inv = {v: k for k, v in self.classes.items()}

        self.stored_in_mem = not read_to_mem
        self.input_transform = input_transform

        image2index = {}
        image_index = 0
        label_data = json.load(open(anno_path, "r", encoding="utf-8"))
        for ann in label_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image2index:
                image2index[image_id] = None
                image_index = len(image2index) - 1
                image2index[image_id] = image_index

                image = None
                for image in label_data["images"]:
                    if image["id"] == image_id:
                        break

                if read_to_mem:
                    self.data["images"] = Image.open(os.path.join(image_dir, image["file_path"]))
                else:
                    self.data["images"].append(image["file_path"])

                self.data["targets"].append([])
            else:
                image_index = image2index[image_id]

            self.data["targets"][image_index].append(self.classes[ann["label"]])

        # make "N-hot" vectors
        for i, d in enumerate(self.data["targets"]):
            temp_tensor = torch.zeros(len(self.classes))
            temp_tensor[d] = 1.0

            self.data["targets"][i] = temp_tensor

    def __getitem__(self, index):
        if self.stored_in_mem:
            input = self.data["images"][index]
        else:
            input = Image.open(os.path.join(self.image_dir, self.data["images"][index]))

        if self.input_transform:
            input = self.input_transform(input)

        return input, self.data["targets"][index]

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.data["images"])

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)


