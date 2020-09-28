# 내용
* train 시 데이터 셋만 만들어주고 cfg.py 파일안에 셋팅값만 변경 해주면됨.
* inference 시 model.py 가장 아래줄 보면됨.

Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```
    
Load pytorch weights (pth file) to do the inference

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```
