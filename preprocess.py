import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


'''
LDPolypVideo dataset
'''
def save_box_ld(path):
    with open(path+'train_box.txt', 'w') as train_box:
        for folder in sorted(os.listdir(path+'Train/Annotations'), key=lambda x:int(x)):
            for text in sorted(os.listdir(path+'Train/Annotations/'+folder)):
                print(folder + '/' + text)
                with open(path+'Train/Annotations/'+folder+'/'+text) as txt:
                    lines = txt.readlines()
                    polyp_num = lines[0].strip()
                    if int(polyp_num) == 0:
                        continue
                    else:
                        i = 1
                        line  = folder+'/'+text.replace('.txt', '.jpg')+';'
                        while(i < len(lines)):
                            xmin, ymin, xmax, ymax = lines[i].strip().split()
                            line = line+xmin+' '+ymin+' '+xmax+' '+ymax+' '
                            i += 1
                        if line[-1]!=';':
                            train_box.write(line[:-1]+'\n')

    with open(path+'test_box.txt', 'w') as test_box:
        for folder in sorted(os.listdir(path+'Test/Annotations'), key=lambda x:int(x)):
            for text in sorted(os.listdir(path+'Test/Annotations/'+folder)):
                print(folder + '/' + text)
                with open(path+'Test/Annotations/'+folder+'/'+text) as txt:
                    lines = txt.readlines()
                    polyp_num = lines[0].strip()
                    if int(polyp_num) == 0:
                        continue
                    else:
                        i = 1
                        line  = folder+'/'+text.replace('.txt', '.jpg')+';'
                        while(i < len(lines)):
                            xmin, ymin, xmax, ymax = lines[i].strip().split()
                            line = line+xmin+' '+ymin+' '+xmax+' '+ymax+' '
                            i += 1
                        if line[-1]!=';':
                            test_box.write(line[:-1]+'\n')

'''
SUN SEG dataset
'''
def save_box_sun(path):
    with open(path+'train_box.txt', 'w') as fbox:
        with open(path+'TrainDataset/Detection/bbox_annotation.json', 'r') as f:
            files = json.load(f)
            for anno in tqdm(files['annotation']):
                idx = anno['id']
                for img in files['images']:
                    if img['id'] == idx:
                        file_name = img['file_name']
                        line = idx.split('-')[0] + '/' + file_name + '.jpg;'
                        break
                ymin, xmin, width, height = anno['bbox']  # the annotation xmin, ymin, xmax, ymax is not correct.
                xmax = xmin + width
                ymax = ymin + height
                line = line + str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                if line[-1] != ';':
                    fbox.write(line[:-1] + '\n')

    with open(path+'test_easy_box.txt', 'w') as fbox:
        with open(path+'TestEasyDataset/Detection/bbox_annotation.json', 'r') as f:
            files = json.load(f)
            for anno in tqdm(files['annotation']):
                idx = anno['id']
                for img in files['images']:
                    if img['id'] == idx:
                        file_name = img['file_name']
                        line = idx.split('-')[0] + '/' + file_name + '.jpg;'
                        break
                ymin, xmin, width, height = anno['bbox']  # the annotation xmin, ymin, xmax, ymax is not correct.
                xmax = xmin + width
                ymax = ymin + height
                line = line + str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                if line[-1] != ';':
                    fbox.write(line[:-1] + '\n')

    with open(path+'test_hard_box.txt', 'w') as fbox:
        with open(path+'TestHardDataset/Detection/bbox_annotation.json', 'r') as f:
            files = json.load(f)
            for anno in tqdm(files['annotation']):
                idx = anno['id']
                for img in files['images']:
                    if img['id'] == idx:
                        file_name = img['file_name']
                        line = idx.split('-')[0] + '/' + file_name + '.jpg;'
                        break
                ymin, xmin, width, height = anno['bbox']  # the annotation xmin, ymin, xmax, ymax is not correct.
                xmax = xmin + width
                ymax = ymin + height
                line = line + str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                if line[-1] != ';':
                    fbox.write(line[:-1] + '\n')

'''
CVC-VideoClinicDB
'''

def save_box_cvc(path):
    with open(path+'train_box2.txt', 'w') as fbox:
        for folder in tqdm(sorted(os.listdir(path+'train/mask'), key=lambda x: int(x))):
            for name in sorted(os.listdir(path+'train/mask/'+folder)):
                mask  = cv2.imread(path+'train/mask/'+folder+'/'+name, cv2.IMREAD_GRAYSCALE)
                H,W = mask.shape
                line  = folder+'/'+name+';'
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                for contour in contours:
                    x,y,w,h = cv2.boundingRect(contour)
                    xmin, ymin, xmax, ymax = max(int(x), 0), max(int(y), 0), min(int(x+w), W-1), min(int(y+h), H-1)
                    line    = line+str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                if line[-1]!=';':
                    fbox.write(line[:-1]+'\n')

    with open(path+'val_box2.txt', 'w') as fbox:
            for folder in tqdm(sorted(os.listdir(path+'validation/mask'), key=lambda x: int(x))):
                for name in sorted(os.listdir(path+'validation/mask/'+folder)):
                    mask  = cv2.imread(path+'validation/mask/'+folder+'/'+name, cv2.IMREAD_GRAYSCALE)
                    H,W = mask.shape
                    line  = folder+'/'+name+';'
                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    for contour in contours:
                        x,y,w,h = cv2.boundingRect(contour)
                        xmin, ymin, xmax, ymax = max(int(x), 0), max(int(y), 0), min(int(x+w), W-1), min(int(y+h), H-1)
                        line    = line+str(int(xmin))+' '+str(int(ymin))+' '+str(int(xmax))+' '+str(int(ymax))+' '
                    if line[-1]!=';':
                        fbox.write(line[:-1]+'\n')

if __name__=='__main__':
    DATAPATH = 'dataset/'   # path to your dataset locations
    if sys.argv[1] == 'ld':
        save_box_ld(DATAPATH)
    if sys.argv[1] == 'sun':
        save_box_sun(DATAPATH)
    if sys.argv[1] == 'cvc':
        save_box_cvc(DATAPATH)