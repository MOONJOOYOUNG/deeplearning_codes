import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import imagefolder

def get_loader(args):
    mean = [0.5, 0.5, 0.5]
    stdv = [0.5, 0.5, 0.5]


    train_transform = transforms.Compose([transforms.Resize((32,32)),
                                          transforms.transforms.RandomCrop(32, padding=8),
                                          transforms.transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=stdv)])

    test_transform = transforms.Compose([transforms.Resize((32,32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=stdv)])

    train_dataset = imagefolder.ImageFolder('/daintlab/data/jooyoung/crop/train/train/', transform=train_transform)
    test_dataset = imagefolder.ImageFolder('/daintlab/data/jooyoung/crop/test/test/', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4)
    if args.remove == 'remove':
        print("Dataset Remove")
        print("Origin Dataset : {0}".format(len(train_loader.dataset)))
        forgetting_history = torch.load(args.forget_histroy)
        sort_mode = args.sort
        if args.mode == 'forgettable':
            examples, filename, _ = forgetting_history.get_forgettable_examples(sorted=sort_mode)
        elif args.mode == 'unforgettable':
            examples, filename, _ = forgetting_history.get_unforgettable_examples(sorted=sort_mode)

        remove_ratio = int((len(examples) / 100 * args.remove_ratio))
        print("Remove count : {0}".format(remove_ratio))
        train_dataset.remove_data(filename[:remove_ratio])
        print("Remove Dataset : {0}".format(len(train_loader.dataset)))

    print("-------------------Make loader-------------------")
    print('Train Dataset :', len(train_loader.dataset),
          '   Test Dataset :', len(test_loader.dataset))

    return train_loader, test_loader