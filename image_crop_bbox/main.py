# 문주영 코드...
import utils
import preprocessing

import os
import time
import argparse

from PIL import Image
from xml.etree.ElementTree import parse

parser = argparse.ArgumentParser(description='Crop Image(read xml files)')
parser.add_argument('--annotation_path', default='./Annotations', type=str, help='Annotation file directory')
parser.add_argument('--image_path', default='./JPEGImages', type=str, help='Image file directory')
parser.add_argument('--save_path', default='./save', type=str, help='Save directory')
args = parser.parse_args()

# python main --annotation_path ./annotation --image_path ./images --save_path ./save
def main():
    # preprocessing
    # 매칭되지 않는 파일 삭제 및 파일명 구조 통일화
    preprocessing.remove_rename(args)

    # save_path check
    utils.check_directory(args.save_path)

    # read files & sorting
    annotation_files = os.listdir(args.annotation_path)
    images_files = os.listdir(args.image_path)

    annotation_files_sort = sorted(annotation_files)
    images_files_sort = sorted(images_files)
    assert (len(annotation_files_sort) != len(images_files_sort), '파일 개수가 맞지 않음 anno : {0}, images : {1}'.format(len(annotation_files),len(images_files)))

    # start
    print('Crop start')
    start = time.time()  # 시작 시간 저장
    crop_image_count = 0
    for i in range(len(images_files)):
        annotation_file = utils.tag_remove_parser(annotation_files_sort[i])
        images_file = utils.tag_remove_parser(images_files_sort[i])

        # .DS_Store : mac에서 발생하는 os 오류.
        if(annotation_file != images_file or annotation_file == '.DSStore' or images_file == '.DSStore'):
             print('파일명이 일치 하지 않음 {0} 번째 파일'.format(i))
             continue
        # read xml, image files
        tree = parse(os.path.join(args.annotation_path, annotation_file + '.xml',))
        origin_image = Image.open(os.path.join(args.image_path, images_file + '.jpg'))
        # read xml
        root = tree.getroot()
        # Find first tag
        elements = root.findall("object")
        # Get Class name
        names = [x.findtext("name") for x in elements]
        # Get annotation
        xmin_list = []; ymin_list = []; xmax_list = []; ymax_list = []
        for element in elements:
            # xml -> object -> bndbox -> [xmin, ymin, xmax, ymax]
            xmin_list.append(int(element.find('bndbox').find('xmin').text)); xmax_list.append(int(element.find('bndbox').find('xmax').text))
            ymin_list.append(int(element.find('bndbox').find('ymin').text)); ymax_list.append(int(element.find('bndbox').find('ymax').text))
        # image crop & save
        for i, name in enumerate(names):
            bndbox_area = (xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
            crop_image = origin_image.crop(bndbox_area)
            crop_image.save(os.path.join(args.save_path, '{0}_{1}_{2}.jpg'.format(images_file, name, i)))
            # image generate counting
            crop_image_count +=1
    print('Crop end')
    print('생성된 이미지 수 :',crop_image_count)
    print("Crop time :", time.time() - start)

    print('File move start')
    start = time.time()  # 시작 시간 저장
    utils.move_files(args.save_path)
    print("Move time :", time.time() - start)
    print('File move end')

if __name__ == "__main__":
    main()
