# Make object detection datasets
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
