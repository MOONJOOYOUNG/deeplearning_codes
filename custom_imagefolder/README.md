import imagefolder
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

mean = [0.5, 0.5, 0.5]
stdv = [0.5, 0.5, 0.5]

test_transform = transforms.Compose([transforms.Resize((800,800)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=stdv)])

test_dataset = imagefolder.ImageFolder('./test', transform=test_transform)


train_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=4)

print(len(test_dataset.samples), len(test_dataset.targets))
test_dataset.remove_data('20200401_161246.jpg')
print(len(test_dataset.samples), len(test_dataset.targets))


for i, (input, target, idx, filename) in enumerate(train_loader):
	print(input, tartget, idx, filename)
