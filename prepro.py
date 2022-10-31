import cv2 as cv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils

def filtro(kernel,img):
    img = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
    return img

def findContours(img):
    cnts = cv.findContours(img.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts



def edgeDetection(img):
    edged = cv.Canny(img, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)
    return edged


def resized(img,scale):
    scale_percent = scale# percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

def prePro(img):
   
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #imagem cinza
   
    img=cv.GaussianBlur(img,(7,7),0) #blur pra melhorar a imagem
    
    return img

