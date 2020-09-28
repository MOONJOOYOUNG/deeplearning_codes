import utils
import numpy as np
import os
import time

def remove_rename(args):
    print('Crop start')
    start = time.time()  # 시작 시간 저장
    # save_path check
    utils.check_directory(args.save_path)

    # read files & sorting
    annotation_files = os.listdir(args.annotation_path)
    images_files = os.listdir(args.image_path)
    annotation_files_sort = sorted(annotation_files)
    images_files_sort = sorted(images_files)
    assert (len(annotation_files_sort) != len(images_files_sort), '파일 개수가 맞지 않음 anno : {0}, images : {1}'.format(len(annotation_files),len(images_files)))

    orgin_ano = []; orgin_img = []
    re_ano = []; re_img = []
    for i in annotation_files_sort:
        annotation_file = utils.tag_remove_parser(i)
        orgin_ano.append(i); re_ano.append(annotation_file);

    for i in images_files_sort:
        images_file = utils.tag_remove_parser(i)
        orgin_img.append(i); re_img.append(images_file)

    orgin_ano = np.array(orgin_ano); orgin_img = np.array(orgin_img)
    re_ano = np.array(re_ano); re_img = np.array(re_img)

    # Remove file
    utils.remove_files(re_ano, re_img, orgin_img, args.image_path)
    utils.remove_files(re_img, re_ano, orgin_ano, args.annotation_path)

    # Modify file name
    for i in annotation_files_sort:
        annotation_file = utils.tag_remove_parser(i)
        os.rename(os.path.join(args.annotation_path,i), os.path.join(args.annotation_path,annotation_file + '.xml'))

    for i in images_files_sort:
        images_file = utils.tag_remove_parser(i)
        os.rename(os.path.join(args.image_path,i), os.path.join(args.image_path,images_file + '.jpg'))

    print("Preprocessing time :", time.time() - start)