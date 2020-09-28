Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```
    
Load pytorch weights (pth file) to do the inference

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```
