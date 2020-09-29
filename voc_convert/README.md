# Make object detection datasets
* 오브젝트 디텍션 데이터셋 제작.
* 이미지와 xml(좌표)파일이 각각 잇을때 밑의 코드 순서대로 진행.
* 같은 폴더에 있다면 서로 경로는 분리시키고 -> class 명만 추출해 data.name파일 제작.
* 이미지의 파일 명을 txt로 만든 후 -> voc 파일로 다시 만들어줌 ex) 이미지경로 좌표 클래스명
## Image & xml converter

### move_image_xml.py
* ./train/...xml & ...jpg -> xml & image files 
* Move xml & image -> ./dataset/train/annotation/...xml , ./dataset/train/image/...jpg
``` 
python move_image_xml.py --path ./dataset/train
``` 

### make_names.py
* data.names -> .dataset/train/data.names
``` 
python make_names.py --path ./dataset/train
``` 

### make_image_txt.py
* image.txt -> .dataset/train/image.txt
``` 
python make_image_txt.py --path ./dataset/train
``` 

### voc_convert.py
* read -> image.txt & ./dataset/train/annotation/...xml
* train.txt -> .dataset/train/train.txt
* file_name (1:train, 2:val, 3:test)
``` 
python voc_convert.py --path ./dataset/train --file_name 1

train.txt
image.path bbox_annotations classname
pytorch_yolov4/data/dataset/train/image/apple_1.jpg 8,15,331,349,0
pytorch_yolov4/data/dataset/train/image/apple_10.jpg 56,99,1413,1419,0
pytorch_yolov4/data/dataset/train/image/apple_11.jpg 213,33,459,258,0 1,30,188,280,0 116,5,337,220,0
``` 
