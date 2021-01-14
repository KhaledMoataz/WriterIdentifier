import cv2
import numpy as np
import os


# ii = "0" + str(i) if i < 10 else str(i)
files = os.listdir("../formsA-D")
for file in files:
# o_img = cv2.imread('data/{}/{}/{}.png'.format(ii, j, k), cv2.IMREAD_GRAYSCALE)
    o_img = cv2.imread('../formsA-D/{}'.format(file), cv2.IMREAD_GRAYSCALE)
    scale_percent = 20
    width = int(o_img.shape[1] * scale_percent / 100)
    height = int(o_img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(o_img, dsize, interpolation=cv2.INTER_AREA)
    img = 255 - img
    ret, thresh = cv2.threshold(img, 40, 255, 0)
    # thresh = cv2.dilate(thresh, np.ones((3, 3)), iterations=1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_img = np.zeros((height, width), dtype=np.uint8)
    top = 0
    bottom = height
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < width / 2:
            continue
        count += 1
        if bottom == height:
            bottom = y if y != height else y - 1
        else:
            top = y + h
            break
        # cv2.rectangle(c_img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # if count != 3:
    #     print("Error ", file, count)
    # img = o_img[int(top * 100 / scale_percent):int(bottom * 100 / scale_percent), :]
    img = img[top:bottom, :]
    cv2.imshow("image", img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
