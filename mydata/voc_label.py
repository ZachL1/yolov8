# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
import argparse
 

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/mnt/d/workspace/yolov8', type=str, help='work dir')
args = parser.parse_args()

sets = ['train', 'val', 'test']
classes = ["car", "person"]   # 改成自己的类别

# base_dir = os.getcwd()
base_dir = args.base_dir
print(base_dir)
 
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h
 
def convert_annotation(image_id):
    in_file = open(f'{base_dir}/mydata/xml/{image_id}.xml', encoding='UTF-8')
    out_file = open(f'{base_dir}/mydata/labels/{image_id}.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            print(f'{image_id} has difficult {difficult} or class {cls} not in classes')
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
wd = getcwd()
for image_set in sets:
    if not os.path.exists(f'{base_dir}/mydata/labels/'):
        os.makedirs(f'{base_dir}/mydata/labels/')
    if not os.path.exists(f'{base_dir}/mydata/paper_data/'):
        os.makedirs(f'{base_dir}/mydata/paper_data/')

    image_ids = open(f'{base_dir}/mydata/dataSet/{image_set}.txt').read().strip().split()
    list_file = open(f'{base_dir}/mydata/paper_data/{image_set}.txt', 'w')
    for image_id in image_ids:
        list_file.write(base_dir + '/mydata/images/%s.JPG\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()