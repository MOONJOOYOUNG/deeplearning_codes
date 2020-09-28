import os
import argparse

from PIL import Image
from xml.etree.ElementTree import parse

parser = argparse.ArgumentParser(description='Grad-Cam-Pytroch')
parser.add_argument('--xml', default='./annot', type=str, help='xml directory')
parser.add_argument('--origin', default='./origin_image', type=str, help='origin image directory')
parser.add_argument('--cam', default='./cam_image', type=str, help='cam image directory')
parser.add_argument('--save', default='./save', type=str, help='save directory')
args = parser.parse_args()

# python cam_crop --xml ./annot --origin origin_image --cam cam_image --save ./save/

xmls = sorted(os.listdir(args.xml))
origin_images = sorted(os.listdir(args.origin))
cam_images = sorted(os.listdir(args.cam))

count = 1
for i in range(len(cam_images)):
    if (cam_images[i] == '.DS_Store'):
        print('파일명이 일치 하지 않음 {0} 번째 파일')
        continue
    print(i,count)
    print(xmls[count],origin_images[count],cam_images[i])
    tree = parse('./annot/' + xmls[count])
    origin_image = Image.open('./origin_image/' + origin_images[count])
    cam_image = Image.open('./resize/' + cam_images[i])

    # read xml
    root = tree.getroot()
    # Find first tag
    elements = root.findall("object")
    # Get Class name
    names = [x.findtext("name") for x in elements]

    xmin_list = []; ymin_list = [];
    xmax_list = []; ymax_list = []
    # Get annotation
    for element in elements:
        # xml -> object -> bndbox -> [xmin, ymin, xmax, ymax]
        xmin_list.append(int(element.find('bndbox').find('xmin').text)); xmax_list.append(int(element.find('bndbox').find('xmax').text))
        ymin_list.append(int(element.find('bndbox').find('ymin').text)); ymax_list.append(int(element.find('bndbox').find('ymax').text))

    # image crop & save
    for j, name in enumerate(names):
        bndbox_area = (xmin_list[j], ymin_list[j], xmax_list[j], ymax_list[j])
        crop_image = cam_image.crop(bndbox_area)
        # paste image
        origin_image.paste(crop_image, (xmin_list[j], ymin_list[j]))

    origin_image.save(os.path.join(args.save,'{0}'.format(cam_images[i])))


