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
    def __init__(self, image_dir, anno_path, input_transform=None,
                 labels_path=None,
                 used_category=-1):

        self.data = {"images": [], "targets": []}
        self.image_dir = image_dir

        image2index = {}
        image_index = 0
        label_data = json.load(open(anno_path, "r", encoding="utf-8"))
        for ann in label_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in image2index:
                image2index[image_id] = image_index
                image_index = len(image2index) - 1

                image = None
                for image in label_data["images"]:
                    if image["id"] == image_id:
                        break

                self.data["images"].append(image["file_path"])
                self.data["targets"].append([])
            else:
                image_index = image2index[image_id]

            self.data["targets"][image_index].append(ann["label"])

    def __getitem__(self, index):
        input = self.coco[index][0]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.labels[index]

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
        return len(self.coco)

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)


