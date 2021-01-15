from FeatureExtractor import FeatureExtractor
from classifier import Classifier
from Utils import utils
import numpy as np
import cv2
import timeit
import preprocessor
import random

random.seed(20)

feature_extractor = FeatureExtractor.FeatureExtractor(3)
classifier = Classifier(1)

preprocessing_time = 0
preprocessed_images_count = 0
feature_extraction_time = 0
training_time = 0
classifying_time = 0


def read_ascii(path):
    f = open(path, "r")
    dataset = [[] for _ in range(672)]
    for line in f:
        if line[0] == '#':
            continue
        img_name, writer, a, b, c, d, e, f = line.split()
        dataset[int(writer)].append(img_name)
    dataset = [writer_dataset for writer_dataset in dataset if len(writer_dataset) >= 2]
    return dataset


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    scale_percent = 15
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_image(images_list, image_name):
    global preprocessing_time, preprocessed_images_count
    base = "../../forms"
    img = read_image("{}/{}.png".format(base, image_name))
    start_time = timeit.default_timer()
    img = preprocessor.preProcessor(img)
    img = img[5:-5, 5:-5]
    # threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img[img > threshold] = 255
    preprocessing_time += timeit.default_timer() - start_time
    preprocessed_images_count += 1
    images_list.append(img)
    return img


def append_features(features, images_list, image_name):
    return np.append(features,
                     feature_extractor.extract_features(preprocess_image(images_list, image_name)), axis=0)


def show_results(truth, predicted, train_images, test_images, train_labels):
    for i, (truth_label, predicted_label) in enumerate(zip(truth, predicted)):
        if truth_label != predicted_label:
            for j, (label, image) in enumerate(zip(train_labels, train_images)):
                if label == truth_label or label == predicted_label:
                    window_x_position = 10 if label == truth_label else 950
                    window_name = "train-{}-{}".format(label, j % 2)
                    cv2.namedWindow(window_name)
                    cv2.moveWindow(window_name, window_x_position, (j % 2) * 350 + 10)
                    cv2.imshow(window_name, image)
            window_name = "test-{}-{}".format(truth_label, predicted_label)
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 500, 200)
            cv2.imshow(window_name, test_images[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()


dataset = read_ascii("../../ascii/forms.txt")
dataset_length = (len(dataset) // 3) * 3
cases_count = 0
test_count = 0
passed_count = 0

while test_count < 1000:
    writers = random.sample(range(0, len(dataset)), 3)
    # writers = [i, i + 1, i + 2]
    train_images_names = []
    train_images = []
    train_labels = []
    test_images_names = []
    test_images = []
    test_labels = []
    for writer in writers:
        pages_count = len(dataset[writer])
        writer_pages = range(0, pages_count)
        pages = random.sample(writer_pages, 2)
        train_images_names += [dataset[writer][pages[0]], dataset[writer][pages[1]]]
        train_labels += [writer] * 2
        writer_pages = set(writer_pages) - set(pages)
        for page in writer_pages:
            test_images_names.append(dataset[writer][page])
        test_labels += [writer] * len(writer_pages)
        # pages_count = len(dataset[writer])
        # train_images_count = min(2, pages_count)
        # train_images += dataset[writer][:train_images_count]
        # train_labels += [writer] * train_images_count
        # writer_test_paths = dataset[writer][2:]
        # test_images += writer_test_paths
        # test_labels += [writer] * len(writer_test_paths)
    if len(test_labels) == 0:
        continue
    cases_count += 1

    start_time = timeit.default_timer()

    features = feature_extractor.extract_features(preprocess_image(train_images, train_images_names[0]))
    for image_name in train_images_names[1:]:
        features = append_features(features, train_images, image_name)
    for image_name in test_images_names:
        features = append_features(features, test_images, image_name)

    feature_extraction_time += timeit.default_timer() - start_time

    test_images_count = len(test_images_names)
    # features = feature_extractor.apply_pca(features)
    classifier.clear()

    start_time = timeit.default_timer()
    classifier.train(features[:-test_images_count], train_labels)
    training_time += timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    predicted = classifier.classify(features[-test_images_count:])
    classifying_time += timeit.default_timer() - start_time

    new_test_count = len(test_labels)
    new_passed_count = new_test_count - np.count_nonzero(predicted - np.array(test_labels))
    test_count += new_test_count
    passed_count += new_passed_count
    print("{}/{}".format(new_passed_count, new_test_count))
    print("Total: {}/{}".format(passed_count, test_count))
    print(passed_count / test_count * 100)
    show_results(test_labels, predicted, train_images, test_images, train_labels)
    print("Average Time:\npreprocessing: {}\nfeature extraction: {}\ntraining: {}\nclassifying: {}\n".format(
        preprocessing_time / preprocessed_images_count,
        feature_extraction_time / preprocessed_images_count,
        training_time / cases_count,
        classifying_time / test_count
    ))
    print()

print("Number of test cases: ", cases_count)
print("Tests count: ", test_count)
print("Passed count: ", passed_count)
print(passed_count / test_count * 100)

# for i in range(1, 10):
#     path = "../data/0" + str(i)
#     features = extract_features(path + "/1/1.png")
#     features = np.append(features, extract_features(path + "/1/2.png"), axis=0)
#     features = np.append(features, extract_features(path + "/2/1.png"), axis=0)
#     features = np.append(features, extract_features(path + "/2/2.png"), axis=0)
#     features = np.append(features, extract_features(path + "/3/1.png"), axis=0)
#     features = np.append(features, extract_features(path + "/3/2.png"), axis=0)
#     features = np.append(features, extract_features(path + "/test.png"), axis=0)
#
#     # cv2.imshow('Original Image', img)
#     # print(type(features))
#     # print(features.shape)
#     # print(features)
#     features = feature_extractor.apply_pca(features)
#     # print(type(features))
#     # print(features.shape)
#     # print(features)
#
#     classifier = Classifier(3)
#     classifier.train(features[:-1], [1, 1, 2, 2, 3, 3])
#     print(classifier.classify([features[-1]]))
#
# start = timeit.default_timer()
# srs_masks = feature_extractor.get_srs_images(img)
# stop = timeit.default_timer()
#
# print('Time: ', stop - start)
# utils.show_images(srs_masks)
#
# cv2.destroyAllWindows()
#
#
#
#
