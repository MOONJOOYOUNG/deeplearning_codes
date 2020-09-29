# Pytorch Custom datasets
* torchvision.datasets.imagefolder을 이용한 customdata set생성
* 기존 데이터셋에서 데이터를 지우는 기능 추가 -> 원본 파일명을 조건으로 삭제 가능
* 기존 로더의 return 값에서 데이터의 index와 filename을 추가로 받아오게 수정

``` 
# Test
test_dataset = imagefolder.ImageFolder('./test', transform=test_transform)
train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
print(len(test_dataset.samples), len(test_dataset.targets))

# filename matching
test_dataset.remove_data('20200401_161246.jpg')
print(len(test_dataset.samples), len(test_dataset.targets))

for i, (input, target, idx, filename) in enumerate(train_loader):
    print(input, target, idx, filename)
``` 

