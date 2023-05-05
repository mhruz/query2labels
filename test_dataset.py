from lib.dataset.naki_multilabel_dataset import NAKIDataset
from torch.utils.data import DataLoader
from dataset.utils import CropAndDeskew, PadImage
from randaugment import RandAugment
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data_transform_list = [CropAndDeskew(),
                             PadImage(),
                             transforms.ToPILImage(),
                             transforms.Resize((1200, 1200)),
                             # RandAugment(),
                             transforms.ToTensor(),
                             normalize
                             ]

train_data_transform = transforms.Compose(train_data_transform_list)

dataset = NAKIDataset("/home/mighty/data/NAKI/NAKI-Classification/images",
                      "/home/mighty/data/NAKI/NAKI-Classification/naki_val.json",
                      read_to_mem=False, input_transform=train_data_transform)

train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

for i, data in enumerate(train_loader):
    # print(i, data)
    plt.imshow(data[0][0].numpy().transpose(1, 2, 0)[:, :, ::-1])
    plt.show()
