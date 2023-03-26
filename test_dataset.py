from lib.dataset.naki_multilabel_dataset import NAKIDataset
from torch.utils.data import DataLoader

dataset = NAKIDataset("/home/mighty/data/NAKI/NAKI-Classification/images",
                      "/home/mighty/data/NAKI/NAKI-Classification/metadata.json")

train_loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0, drop_last=False)

for i, data in enumerate(train_loader):
    print(i, data)
