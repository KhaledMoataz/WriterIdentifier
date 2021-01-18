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


def show_results(truth, predicted, train_images, test_images, train_labels):
    shown = False
    for i, (truth_label, predicted_label) in enumerate(zip(truth, predicted)):
        if truth_label != predicted_label:
            for j, (label, (image_name, image)) in enumerate(zip(train_labels, train_images)):
                if label == truth_label or label == predicted_label:
                    window_x_position = 10 if label == truth_label else 950
                    window_name = "{}-train-{}-{}".format(image_name, label, j % 2)
                    cv2.namedWindow(window_name)
                    cv2.moveWindow(window_name, window_x_position, (j % 2) * 350 + 10)
                    cv2.imshow(window_name, image)
                    shown = True
            window_name = "{}-test-{}-{}".format(test_images[i][0], truth_label, predicted_label)
            print("{} {} {}".format(test_images[i][0], truth_label, predicted_label))
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 500, 200)
            cv2.imshow(window_name, test_images[i][1])
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
    return shown
