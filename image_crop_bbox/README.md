# Matching image & annotataion, Crop Image to annotation
## 이미지 파일과 좌표를 가지고 새로운 이미지 파일 생성.
* 파일명 구조화 및 매칭 되지 않는 파일 삭제
* 파일 이름 매칭 후 바운딩 박스 좌표 기준으로 이미지 crop
* 클래스별로 해당 폴더에 저장

## main.py
``` 
# python main --annotation_path ./annotation --image_path ./images --save_path ./save
``` 
