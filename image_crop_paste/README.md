# Pytorch Grad-Cam

## main.py
* cam image에서 bbox 좌표를 가지고 이미지를 자른 후 원본 이미지에 복사
``` 
python cam_crop --xml ./annot --origin origin_image --cam cam_image --save ./save/
``` 

![example](./example.png)
