# Pytorch Custom datasets
* torchvision.datasets.imagefolder

``` 
test_dataset = imagefolder.ImageFolder('./test', transform=test_transform)
train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
print(len(test_dataset.samples), len(test_dataset.targets))

test_dataset.remove_data('20200401_161246.jpg')
print(len(test_dataset.samples), len(test_dataset.targets))

for i, (input, target, idx, filename) in enumerate(train_loader):
    print(input, target, idx, filename)
``` 

