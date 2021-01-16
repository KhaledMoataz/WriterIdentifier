import cv2


def read_image(path, scale=15):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Reduce the image size
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def show_images(images, names=None):
    for i in range(len(images)):
        name = str(i + 1)
        if names is not None:
            name = names[i]
        cv2.imshow(name, images[i])
    cv2.waitKey(0)
