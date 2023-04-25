import argparse
import json
from math import floor
from naki_multilabel_dataset import NAKIDataset
import os
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split the dataset into individual subsets (tran/val/test)')
    parser.add_argument('dataset_path', type=str, help='path to json dataset')
    parser.add_argument('--train_split', type=float, help='portion of train samples (0-1)', default=0.8)
    parser.add_argument('--test_split', type=float, help='portion of test samples (0-1)', default=0.1)
    parser.add_argument('--val_split', type=float, help='portion of validation samples (0-1)', default=0.1)
    args = parser.parse_args()

    assert args.train_split + args.test_split + args.val_split == 1

    data = json.load(open(args.dataset_path, "r", encoding="utf8"))

    num_samples = len(data["images"])

    train_samples = floor(args.train_split * num_samples)
    test_samples = floor(args.test_split * num_samples)
    val_samples = floor(args.val_split * num_samples)

    # add the remaining data to train set
    train_samples += num_samples - train_samples - test_samples - val_samples

    print(f"Total samples = {num_samples}, will be divided:")
    print(f"\t...train samples = {train_samples}")
    print(f"\t...test samples = {test_samples}")
    print(f"\t...val samples = {val_samples}")

    indexes = [*range(num_samples)]
    random.shuffle(indexes)

    train_indexes = indexes[:train_samples]
    test_indexes = indexes[train_samples:train_samples + test_samples]
    val_indexes = indexes[train_samples + test_samples:]

    print("Extracted samples:")
    print(f"\t...train samples = {len(train_indexes)}")
    print(f"\t...test samples = {len(test_indexes)}")
    print(f"\t...val samples = {len(val_indexes)}")

    train_image_ids = []
    test_image_ids = []
    val_image_ids = []

    train_data = {"images": [], "annotations": []}
    test_data = {"images": [], "annotations": []}
    val_data = {"images": [], "annotations": []}

    for idx, im in enumerate(data["images"]):
        if idx in train_indexes:
            train_image_ids.append(im["id"])
            train_data["images"].append(im)
        elif idx in test_indexes:
            test_image_ids.append(im["id"])
            test_data["images"].append(im)
        else:
            val_image_ids.append(im["id"])
            val_data["images"].append(im)

    for ann in data["annotations"]:
        if ann["image_id"] in train_image_ids:
            train_data["annotations"].append(ann)
        elif ann["image_id"] in test_image_ids:
            test_data["annotations"].append(ann)
        else:
            val_data["annotations"].append(ann)

    dataset_name = os.path.splitext(args.dataset_path)[0]
    train_filename = dataset_name + "_train.json"
    test_filename = dataset_name + "_test.json"
    val_filename = dataset_name + "_val.json"

    with open(train_filename, "w", encoding="utf8") as fp:
        fp.write(json.dumps(train_data))

    with open(test_filename, "w", encoding="utf8") as fp:
        fp.write(json.dumps(test_data))

    with open(val_filename, "w", encoding="utf8") as fp:
        fp.write(json.dumps(val_data))
