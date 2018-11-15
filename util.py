import csv
import numpy as np


def get_train_data(filename = "train.csv"):
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    features = {"y": [], "x": []}
    with open(filename, 'r') as train_file:
        image_reader = csv.reader(train_file, delimiter=',')
        for row in image_reader:
            features["y"].append([row[0]])
            features["x"].append([row[1::]])
        features["x"] = np.array(features["x"])
        features["y"] = np.array(features["y"])
        print("Shape X:", features["x"].shape)
        print("Shape Y:", features["y"].shape)
    return features
def get_test_data(filename = "test.csv"):
    features = {"x": []}
    with open(filename, 'r') as train_file:
        image_reader = csv.reader(train_file, delimiter=',')
        for row in image_reader:
            features["x"].append([row])
        features["x"] = np.array(features["x"])
        print("Shape X:", features["x"].shape)
#make functions to build the csv to make submissions
def make_submission(result_dict, filename="submission.csv"):
    with open(filename, 'w') as submission_file:
        for row in result_dict:
            submission_file.write(row)


if __name__ == '__main__':
    get_train_data()
    get_test_data()


