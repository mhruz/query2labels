import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from dataset.naki_multilabel_dataset import NAKIDataset
from dataset.utils import CropAndDeskew, PadImage
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    if args.dataname == "naki":
        train_data_transform_list = [CropAndDeskew(),
                                     PadImage(),
                                     transforms.Resize((args.img_size, args.img_size)),
                                     RandAugment(),
                                     transforms.ToTensor(),
                                     normalize]
    else:
        train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                     RandAugment(),
                                     transforms.ToTensor(),
                                     normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    if args.dataname == "naki":
        test_data_transform = transforms.Compose([
            CropAndDeskew(),
            PadImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize])
    else:
        test_data_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize])

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
    elif args.dataname == 'naki':
        dataset_dir = args.dataset_dir
        train_dataset = NAKIDataset(
            image_dir=osp.join(dataset_dir, 'images'),
            anno_path=osp.join(dataset_dir, 'naki_train.json'),
            input_transform=train_data_transform
        )
        val_dataset = NAKIDataset(
            image_dir=osp.join(dataset_dir, 'images'),
            anno_path=osp.join(dataset_dir, 'naki_val.json'),
            input_transform=train_data_transform
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset))
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
