import cv2
import math
import numpy as np

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


def getLines(gray):
    # Apply edge detection
    edges = cv2.Canny(gray, 80, 120)
    #cv2.imshow('edges',edges)
    #get horizaontal lines with min length = 80
    lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 60, 1)
    #filter the lines to get only the 3 needed horizontal lines
    filtered_lines = np.empty((0,4),int)
    threshold = 8
    for line1 in lines.reshape(lines.shape[0],4):
        flag = True
        for line2 in filtered_lines:
            # if the line is very close to another exist line on the hight, ignore it
            if(abs(line2[1] - line1[1]) <= threshold):
                flag = False
                break
        if(flag):
            #print(line1)
            filtered_lines = np.vstack([filtered_lines,line1])
            pt1 = (line1[0],line1[1]) #(x1,y1)
            pt2 = (line1[2],line1[3]) #(x2,y2)
            #cv2.imshow('lines',cv2.line(gray, pt1, pt2, (0,0,255), 3))
    
    # Sort the lines descending on the y-axis
    filtered_lines.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    return filtered_lines


def handleLines(img,lines):
    threshold = 3
    # Crop the image to the paragraph with the white parts 
    if(len(lines) == 3):
        y1 = lines[1][1]+threshold
        y2 = lines[2][1]-threshold
    elif(len(lines) == 2):
        if(lines[1][1] > img.shape[0]/2):   # The second line is after the paragraph
            y1 = lines[0][1]+threshold
            y2 = lines[1][1]-threshold
        else:   # both lines above the paragraph
            y1 = lines[1][1]+threshold
            y2 = img.shape[0]
    elif(len(lines) == 1):
        if(lines[0][1] > img.shape[0]/2):   # The line is above the paragraph
            y1 = lines[0][1]+threshold
            y2 = img.shape[0]
        else:       # The line is below the paragraph
            y1 = 0
            y2 = lines[0][1]+threshold
    else:
        print("Error in preProcessing ! number of horizontal lines = %d" %len(lines))
        return gray,1    

    no_lines_image = img[y1:y2,0:img.shape[1]]
    #cv2.imshow('Image with no lines',no_lines_image)
    return no_lines_image,0

def cropPargraph(image):
    # remove the noise that affect on the contours
    blur = cv2.GaussianBlur(image, (3, 3), 0)

    ret, bin_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('',bin_img)
    #Because Some version return 2 parameters and other return 3 parameters
    major = cv2.__version__.split('.')[0]
    if major == '3': img2, contours, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else: contours, hierarchy= cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x_min = 100000
    x_max = -1
    y_min = 100000
    y_max = -1
    for contour in contours:    
        [x,y, w, h] = cv2.boundingRect(contour)
        
        # Special Case: ignore the border contours
        if(h > image.shape[0]/2):
            continue
        if(w > image.shape[1]/2):
            continue

        if(x < x_min):
            x_min = x
        if(x + w > x_max):
            x_max = x + w
        if(y < y_min):
            y_min = y
        if(y + h > y_max):
            y_max = y + h

    paragraph_image = image[y_min:y_max:,x_min:x_max]
    #print(x_min,x_max,y_min,y_max)
    #cv2.drawContours(no_lines_image, contours, -1, (0,255,0), 3)
    #cv2.imshow('Paragraph',paragraph_image)
    return paragraph_image


def preProcessing(path):
    
    gray = readImage(path)
    
    lines = getLines(gray)

    no_lines_image,error = handleLines(gray,lines)

    paragraph = cropPargraph(no_lines_image)

    return  paragraph,error

def testPreProcessing():
    for i in range(1,10):
        for j in range(1,4):
            print(i)
            image,x = preProcessing('../data/0'+str(i)+'/'+str(j)+'/1.png')
            image,x = preProcessing('../data/0'+str(i)+'/'+str(j)+'/2.png')
        image,x = preProcessing('../data/0'+str(i)+'/test.png')

    for j in range(1,4):
        print(j)
        image,x = preProcessing('../data/10/'+str(j)+'/1.png')
        image,x = preProcessing('../data/10/'+str(j)+'/2.png')
    image,x = preProcessing('../data/10/test.png')

testPreProcessing()

#cv2.imshow('final',image)

cv2.waitKey(0)
cv2.destroyAllWindows()