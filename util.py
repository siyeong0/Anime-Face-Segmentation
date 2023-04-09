import cv2 as cv
import glob
import numpy as np
import os

COLOR_BACKGROUND = (0,255,255)
COLOR_HAIR = (255,0,0)
COLOR_EYE = (0,0,255)
COLOR_MOUTH = (255,255,255)
COLOR_FACE = (0,255,0)
COLOR_SKIN = (255,255,0)
COLOR_CLOTHES = (255,0,255)
PALETTE = [COLOR_BACKGROUND,COLOR_HAIR,COLOR_EYE,COLOR_MOUTH,COLOR_FACE,COLOR_SKIN,COLOR_CLOTHES]

def img2seg(path):
    src = cv.imread(path)
    src = src.reshape(-1, 3)
    seg_list = []
    for color in PALETTE:
        seg_list.append(np.where(np.all(src==color, axis=1), 1.0, 0.0))
    dst = np.stack(seg_list,axis=1).reshape(512,512,7)
    
    return dst.astype(np.float32)

def seg2img(src):
    src = np.moveaxis(src,0,2)
    dst = [[PALETTE[np.argmax(val)] for val in buf]for buf in src]
    
    return np.array(dst).astype(np.uint8)