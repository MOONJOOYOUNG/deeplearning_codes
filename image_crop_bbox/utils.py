import os
import shutil
import numpy as np

# 매칭되지 않는 파일 삭제
def remove_files(list_a, list_b, origin_b, path):
    delete_num = []
    for i in list_a:
        idx = np.where(list_b==i)[0]
        delete_num.extend(idx)
    print('Delete files : ',len(delete_num))
    idx = np.array(delete_num)
    delete_li = np.delete(origin_b, idx)
    for i in delete_li:
        os.remove(os.path.join(path, i))

def move_files(save_path):
    # read files
    files = os.listdir(save_path)
    for file in files:
        split_name = file.split('_')
        class_name = split_name[1]
        check_directory(class_name)
        # move(src, dir)
        shutil.move(os.path.join(save_path, file), os.path.join('./' + class_name, file))

def tag_remove_parser(string):
    """
        Args:
            string (str)
        Returns:
            string (str)
    """
    string = string.replace('_', '')
    string = string.replace('.xml', '')
    string = string.replace('.jpg', '')
    string = string.replace('.JPG', '')
    string = string.replace('.jpge', '')
    string = string.replace('.png', '')

    return string

def check_directory(path):
    """
        Args:
            path (str)
    """
    if type(path) != str: return

    if not os.path.exists(path):
        os.makedirs(path)