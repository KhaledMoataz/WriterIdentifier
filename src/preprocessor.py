import cv2
import math
import numpy as np
import os

def readImage(path):
    # Load an color image in grayscale
    img = cv2.imread(path)
    # Reduce the image size 
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # convert the image to gray scale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',gray)
    
    return gray


def betweenLines(gray):
    gray2 = 255 - gray
    ret, bin_img = cv2.threshold(gray2,40,255,cv2.THRESH_BINARY)
    #cv2.imshow('',bin_img)
    #Because Some version return 2 parameters and other return 3 parameters
    major = cv2.__version__.split('.')[0]
    if major == '3': img2, contours, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else: contours, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #lines = 0
    y_min = 0
    y_max = gray.shape[0]
    for contour in contours:    
        [x,y, w, h] = cv2.boundingRect(contour)
        if(w < gray.shape[1]/2):
            continue
        #print(y)
        #lines = lines + 1
        if(y > gray.shape[0]/2 and y < y_max):
            y_max = y
        elif(y < gray.shape[0]/2 and y > y_min):
            y_min = y
    
    #if(lines != 3):
    #    print("Number of lines in preprocessor = %d"%lines)        
    #print(x_min,x_max,y_min,y_max)
    no_lines_image = gray[y_min:y_max,:]
    #print(x_min,x_max,y_min,y_max)
    #cv2.drawContours(gray, contours, -1, (0,255,0), 3)
    #cv2.imshow('Image without lines',no_lines_image)
    return no_lines_image

def cropPargraph(image):
    # remove the noise that affect on the contours
    blur = cv2.GaussianBlur(image, (3, 3), 0)

    ret, bin_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('',bin_img)
    #Because Some version return 2 parameters and other return 3 parameters
    major = cv2.__version__.split('.')[0]
    if major == '3': img2, contours, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else: contours, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x_min = image.shape[1]
    x_max = 0
    y_min = image.shape[0]
    y_max = 0
    thin_width = 10
    thin_hight = 10
    for contour in contours:    
        [x,y, w, h] = cv2.boundingRect(contour)
        
        # Special Case: ignore the border contours
        if(h > image.shape[0]/2 or w > image.shape[1]/2 or w < thin_width or h < thin_hight):
            continue
        
        if(x < x_min):
            x_min = x
        if(x + w > x_max):
            x_max = x + w
        if(y < y_min):
            y_min = y
        if(y + h > y_max):
            y_max = y + h
    
    threshold = 10
    x_min = max(x_min-threshold,0)
    y_min = max(y_min-threshold,0)
    x_max = min(x_max+threshold,image.shape[1])
    y_max = min(y_max+threshold,image.shape[0])
    #print(x_min,x_max,y_min,y_max)
    paragraph_image = image[y_min:y_max:,x_min:x_max]
    #print(x_min,x_max,y_min,y_max)
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #cv2.imshow('Paragraph',paragraph_image)
    return paragraph_image


def preProcessor(path):
    gray = readImage(path)    
    no_lines_image = betweenLines(gray)
    paragraph_image = cropPargraph(no_lines_image)
    return  paragraph_image


def testPreProcessing():
    for i in range(1,10):
        for j in range(1,4):
            print(i)
            image = debugPreProcessor('../data/0'+str(i)+'/'+str(j)+'/1.png')
            cv2.imshow(str(i)+'/'+str(j)+'/1',image)
            image = debugPreProcessor('../data/0'+str(i)+'/'+str(j)+'/2.png')
            cv2.imshow(str(i)+'/'+str(j)+'/2',image)
        print("test")
        image = debugPreProcessor('../data/0'+str(i)+'/test.png')
        cv2.imshow(str(i)+'/test',image)
    for j in range(1,4):
        print(j)
        image = debugPreProcessor('../data/10/'+str(j)+'/1.png')
        cv2.imshow('10/'+str(j)+'/1',image)
        image = debugPreProcessor('../data/10/'+str(j)+'/2.png')
        cv2.imshow('10/'+str(j)+'/2',image)
    print("test")
    image = debugPreProcessor('../data/10/test.png')
    cv2.imshow('10/test',image)
     
def testForms(directory):
    for filename in os.listdir(directory):
        image = preProcessor(directory+filename)

#testPreProcessing()
#testForms('../data/formsA-D/')
#testForms('../data/formsE-H/')
#testForms('../data/formsI-Z/')

#image = preProcessor('../data/formsA-D/a03-089.png')
#image = preProcessor('../data/formsA-D/d06-063.png')
#image = preProcessor('../data/formsA-D/b03-109.png')
#image = preProcessor('../data/formsA-D/a06-157.png')
#image = preProcessor('../data/formsA-D/b04-208.png')
#image = preProcessor('../data/formsE-H/g04-108.png')
#image = preProcessor('../data/formsE-H/h07-087.png')
#image = preProcessor('../data/formsI-Z/r03-115.png')
#image = preProcessor('../data/formsI-Z/p03-189.png')
#image = preProcessor('../data/formsI-Z/m04-251.png')
#image = preProcessor('../data/formsE-H/h02-037.png')
#image = preProcessor('../data/formsI-Z/k01-051.png')
#image = preProcessor('../data/formsI-Z/p02-155.png')
#image = preProcessor('../data/formsI-Z/p02-109.png')

#cv2.imshow('',image)

cv2.waitKey(0)
cv2.destroyAllWindows()