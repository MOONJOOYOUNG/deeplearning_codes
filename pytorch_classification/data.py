import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loader(args):
    mean = [0.5, 0.5, 0.5]
    stdv = [0.5, 0.5, 0.5]


    train_transform = transforms.Compose([transforms.Resize((128,128)),
                                          transforms.transforms.RandomCrop(128, padding=8),
                                          transforms.transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=stdv)])

    test_transform = transforms.Compose([transforms.Resize((128,128)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=stdv)])

    train_dataset = datasets.ImageFolder('/project_ood/dataset/train/train/', transform=train_transform)
    test_dataset = datasets.ImageFolder('/project_ood/dataset/origin/images/', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4)

    print("-------------------Make loader-------------------")
    print('Train Dataset :', len(train_loader.dataset),
          '   Test Dataset :', len(test_loader.dataset))

    return train_loader, test_loader