from FeatureExtractor import FeatureExtractor
from Utils import utils
import preprocessor

import numpy as np
import cv2
import timeit
from matplotlib import pyplot as plt

#start = timeit.default_timer()
#srs_masks = feature_extractor.get_srs_images(img)
#stop = timeit.default_timer()

#print('Time: ', stop - start)

def read_ascii(path):
    f = open(path, "r")
    training = [ [] for _ in range(672) ]
    test = [ [] for _ in range(672) ]
    for line in f:
        if(line[0] == '#'):
            continue
        img_name,writer,a,b,c,d,e,f = line.split()
        
        writer = int(writer)
        if(img_name[0] > 'H'):
            training[writer].append(img_name)
        else:
            test[writer].append(img_name)
    return training,test

def main():
    images_path = "../data/forms/"
    radius = 5
    #####       read ascii file     #####
    ascii_path = "../data/ascii/forms.txt"
    ascii_training,ascii_test = read_ascii(ascii_path)
    print("Done Reading ascii")
    #####       random writers      #####
    writer1 = np.random.randint(672)
    writer2 = np.random.randint(672)
    writer3 = np.random.randint(672)
    while(writer2 == writer1):
        writer2 = np.random.randint(672)
    while(writer3 == writer2 or writer3 == writer1):
        writer3 = np.random.randint(672)
    writers = [writer1,writer2,writer3]
    #print(writers)
    print("Done radnom writers")
    #####       Test Case           #####
    training_list = list()
    label_list = list()
    for writer in writers:
        img1_index = np.random.randint(len(ascii_training[writer])) 
        #print(ascii_training[writer][img1_index])
        img1 = cv2.imread(images_path+ascii_training[writer][img1_index]+".png",0)
        training_list.append(img1)
        label_list.append(writer)

        if(len(ascii_training[writer]) > 1):
            img2_index = np.random.randint(len(ascii_training[writer])) 
            while(img2_index == img1_index):
                img2_index = np.random.randint(len(ascii_training[writer])) 
            img2 = cv2.imread(images_path+ascii_training[writer][img2_index]+".png",0)
            training_list.append(img2)
            label_list.append(writer)
    if(len(ascii_test[writers[0]]) == 0 and len(ascii_test[writers[1]]) == 0 and len(ascii_test[writers[2]]) == 0):
        print("Bad Luck :(")
    print("Done Choose Writers images")
    #####       preprocessing       #####
    for i in range(len(training_list)):
        #print(training_list)
        training_list[i] = preprocessor.preprocess(training_list[i])    
    print("Done preprocessing")
    #####       Apply LBP           #####
    feature_extractor = FeatureExtractor.FeatureExtractor(radius)
    features_list = feature_extractor.extract_features(training_list[0])    
    for i in range(1,len(training_list)):
        features_list = np.append(features_list,feature_extractor.extract_features(training_list[i]),axis=0)    
    print("Done LBP")
    #####       Choose Test Case    #####
    random_writer = np.random.randint(len(writers))
    random_writer = writers[random_writer]
        ## Need to be handled Later
    while(len(ascii_test[random_writer]) < 1):
        random_writer = np.random.randint(len(writers))
        random_writer = writers[random_writer]
    
    random_image = np.random.randint(len(ascii_test[random_writer]))
    random_image = ascii_test[random_writer][random_image]
    
    test_img = cv2.imread(images_path+random_image+".png",0)
    test_img = preprocessor.preprocess(test_img)
    features_list = np.append(features_list,feature_extractor.extract_features(test_img),axis=0)
    print("Done Choosing a Test Case")
    #####       Apply PCA           #####
    pca_features = feature_extractor.apply_pca(features_list)
    print("Done PCA")
    
    #####       Classification      #####
    classifier = Classifier()
    classifier.train(pca_features[:-1], label_list)
    print(classifier.classify([pca_features[-1]]))
    print(random_writer)
main()
#cv2.waitKey(0)
cv2.destroyAllWindows()