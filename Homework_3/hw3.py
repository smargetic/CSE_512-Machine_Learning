import csv
import numpy as np
from scipy.spatial import distance
import operator
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import math


def read_data(fileName):
    data = []
    label = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
            else:
                label.append(row[0])
                data.append(row[1::]) #double check this

    data = np.array(data).astype(np.float64)
    #normalize
    data = data/255.0
    label = np.array(label).astype(int)

    return data, label

def knn_classifier(Xtrain, ytrain, Xtest, k, distance_type = 'euclidean'):
    idxs = []
    y = []
    #go through test data and find x train with least distance
    #distance_matrix = distance.cdist(Xtrain, Xtest, 'euclidean')
    distance_matrix = distance.cdist(Xtrain, Xtest, distance_type)
    distance_matrix = np.array(distance_matrix)
    #each row = test data pt
    distance_matrix = distance_matrix.T

    for i in range(0,len(distance_matrix)):
    #for i in range(0,1):
        distM_temp = list(enumerate(distance_matrix[i]))
        sorted_list = sorted(distM_temp, key=operator.itemgetter(1))
        #get bottom k values (least distance)
        temp_indx = [sorted_list[i][0] for i in range(0,k)]
        idxs.append(temp_indx)

    for i in range(0,len(idxs)):
        temp = []
        #get labels
        for j in range(0,len(idxs[i])):
            temp.append(ytrain[idxs[i][j]])

        #find majority label
        c = Counter(temp)
        value, count = c.most_common()[0]
        #majority is return value
        y.append(value)

    idxs = np.array(idxs)
    y = np.array(y)
    return y, idxs


################################################################## GRAPHING #############################################################################
def plot_accuracy(accuracy, k, x_label, title_):
    plt.plot(k, accuracy)
    plt.title(title_)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.show()

#nearest neighbors
def plot_nn(nn_list, x_train, labels_train, actual_label, predicted_label):
    fig = plt.figure()
    count = 1
    middle = len(nn_list[0])//2
    for i in range(0,len(nn_list)):
        for j in range(0,len(nn_list[i])):
            data = x_train[nn_list[i][j]]

            plt.subplot(len(nn_list),len(nn_list[0]), count)
            plt.imshow(np.reshape(data, (28,28)))
            plt.title(labels_train[nn_list[i][j]])
            if(j==0):
                y_label = "Predicted Label: " + str(predicted_label[i]) + "\nActual Label: " + str(actual_label[i])
                plt.text(x = -45, y = 10, s = y_label, rotation = 0, va = "center")
            plt.axis("off")
            count = count +1
    plt.tight_layout()
    plt.show()

    

################################################################# FUNCTION CALLS #######################################################################
def call_():
    data_train, label_train = read_data("mnist_train.csv")
    data_test, label_test = read_data("mnist_test.csv")
    knn_classifier(data_train, label_train, data_test, 5)

def question_2_1():
    data_train, label_train = read_data("mnist_train.csv")
    data_test, label_test = read_data("mnist_test.csv")
    m = [0,1,2,4,8,16,32]
    k_list = []
    accuracy_list = []
    for i in range(0,len(m)):
        k = (2*m[i]) + 1
        k_list.append(k)
        y_pred, idxs = knn_classifier(data_train, label_train, data_test, k)
        accuracy_list.append(accuracy_score(label_test, y_pred))

    #plot
    plot_accuracy(accuracy_list, k_list, "K Values", "Accuracy Against K")

def question_2_2():
    data_train, label_train = read_data("mnist_train.csv")
    data_test, label_test = read_data("mnist_test.csv")   
    k = 3
    n = [100, 200, 400, 600, 800, 1000]
    accuracy_list = []
    for i in range(0,len(n)):
        y_pred, idxs = knn_classifier(data_train[:n[i],:], label_train[:n[i]], data_test, k)
        accuracy_list.append(accuracy_score(label_test, y_pred))

    #plot
    plot_accuracy(accuracy_list, n, "N Values", "Accuracy Against N")

def question_2_3():
    data_train, label_train = read_data("mnist_train.csv")
    data_test, label_test = read_data("mnist_test.csv") 
    k = 3 

    #euclidean   
    y_pred_euc, _ = knn_classifier(data_train, label_train, data_test, k)
    accuracy_euc = accuracy_score(label_test, y_pred_euc)

    #manhattan   
    y_pred_man, _ = knn_classifier(data_train, label_train, data_test, k, 'cityblock')
    accuracy_man = accuracy_score(label_test, y_pred_man)

    print("Euclidean Distance Accuracy: " + str(accuracy_euc))
    print("Manhattan Distance Accuracy: " + str(accuracy_man))


def question_2_4():
    data_train, label_train = read_data("mnist_train.csv")
    data_test, label_test = read_data("mnist_test.csv") 
    k = 5 
    #predict data pts
    y_pred, idxs = knn_classifier(data_train, label_train, data_test, k)
    
    list_wrong = []
    actual_label = []
    predicted_label = []
    for i in range(0,len(label_test)):
        #wrong classification
        if(label_test[i]!= y_pred[i]):
            actual_label.append(label_test[i])
            predicted_label.append(y_pred[i])
            list_wrong.append(idxs[i])
        if(len(list_wrong)==3):
            break
    plot_nn(list_wrong, data_train, label_train, actual_label, predicted_label)

call_()
#question_2_1()
#question_2_2()
#question_2_3()
#question_2_4()