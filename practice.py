import os
import glob
import cv2
import numpy as np
from collections import OrderedDict
import pickle

def main():
    train_path = "./mnist/train/"
    test_path = "./mnist/test/"

    train_paths = glob.glob(train_path + '/*/*')
    test_paths = glob.glob(test_path + '/*/*')

    train_dataset = read_image_and_label(train_paths)
    test_dataset = read_image_and_label(test_paths)

    save_npy(train_dataset, test_dataset)

    data_dict = read_npy()

    save_pickle(data_dict)

    image = data_dict['train_image'][0]

    data_augment(image)

def read_image_and_label(paths):
    images = []
    labels = []

    for path in paths:
        label = os.path.basename(os.path.dirname(path))
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(image)
        labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

def save_npy(train_dataset, test_dataset):
    train_images, train_labels = train_dataset
    test_images, test_labels = test_dataset

    np.save("./train_images.npy", train_images)
    np.save("./test_images.npy", test_images)
    np.save("./train_labels.npy", train_labels)
    np.save("./test_labels.npy", test_labels)

def read_npy():
    data_dict = OrderedDict()
    
    data_dict['train_image'] = np.load("./train_images.npy")
    data_dict['train_label'] = np.load("./train_labels.npy")
    data_dict['test_image'] = np.load("./test_images.npy")
    data_dict['test_label'] = np.load("./test_labels.npy")
    
    return data_dict

def save_pickle(data_dict):
    with open("./data_dict.pkl", 'wb') as file:
        pickle.dump(data_dict, file)

def data_augment(image):
    cv2.imwrite("./original.jpg", image)

    flipped_image = cv2.flip(image, 1)
    cv2.imwrite("./flipped.jpg", flipped_image)

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("./rotated.jpg", rotated_image)

    resized_image = cv2.resize(image, (100, 100))
    cv2.imwrite("./resized.jpg", resized_image)

if __name__ == "__main__":
    main()
