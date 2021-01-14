import cv2


def show_images(images, names=None):
    for i in range(len(images)):
        name = str(i + 1)
        if names is not None:
            name = names[i]
        cv2.imshow(name, images[i])
    cv2.waitKey(0)
