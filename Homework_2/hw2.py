import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import random
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import statistics

k = 0 #number of classes
loss_list = []

#read data
def read_data(fileName):
    data = []
    death = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
            else:
                data.append(row[:-1]) #double check this
                death.append(row[-1])

    data = np.array(data)
    data = data.astype(np.float64)

    death = np.array(death)
    death = death.astype(int)

    #data = data.T #because xi is column vector

    return data, death

# normalize train and test data
def normalize(train_data, test_data):
    train_data_copy = copy.deepcopy(train_data)
    test_data_copy = copy.deepcopy(test_data)

    #transpose so norm on each column
    #train_data_copy = train_data_copy.T
    #test_data_copy = test_data_copy.T

    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(train_data_copy) #variance is population variance
    transformed_test = scaler.transform(test_data_copy) #only transform test data

    #return to normal orientation
    #transformed_train = np.array(transformed_train.T)
    #transformed_test = np.array(transformed_test.T)
    transformed_train = np.array(transformed_train)
    transformed_test = np.array(transformed_test)

    return transformed_train, transformed_test

#find how many classes there are given output
def find_classes(y):
    classes = []
    for i in range(0,len(y)):
        if(y[i] not in classes):
            classes.append(y[i])

    return classes

#x_bar_batch is a batch of x_bar data points, in row format
def gradient_descent(x_bar_batch, y, theta, k, n):
    #for each class
    for i in range(0,k-1):
        sum = 0 # should change something

        for j in range(0, len(x_bar_batch)):
            prob_ = prob(i+1,x_bar_batch[j],theta, k)

            delta = 0
            if(y[j] == (i+1)): #if the class and output line up
                delta = 1
            

            inner = (delta - prob_) #not sure yet if p is a vector ==> NEED TO STILL MULTIPLY BY 
            sum = sum + (inner * x_bar_batch[j])
        deriv = -(1/len(x_bar_batch))*sum

        theta[:,i] = theta[:,i] - (n*deriv)
            

    return theta

#calculated prob of x_i belonging to class class_
#x_i is in row format
def prob(class_, x_i, theta, k):
    num = 1
    x_i_temp = copy.deepcopy(x_i.T)
    #a = 0
    array_temp = []
    for i in range(0,k-1):
        theta_temp = copy.deepcopy(theta[:,i])
        x_i_temp = copy.deepcopy(x_i.T)
        array_temp.append(np.matmul(theta_temp, x_i_temp))

    #a = np.min(np.array(array_temp))
    a = np.max(np.array(array_temp))
    #a=statistics.median(np.array(array_temp))

    if(class_ != 0): #when class ==0 num =1
        theta_temp = copy.deepcopy(theta[:,class_ -1]) #becomes row vector
        x_i_temp = copy.deepcopy(x_i.T)
        num = np.exp(np.matmul(theta_temp, x_i_temp) - a) #SHOULD BE SCALAR
    else:
        num = np.exp(-a)
        #num = np.exp(np.exp(np.log(theta_temp) + np.log(x_i_temp))) #SHOULD BE SCALAR

    denom = np.exp(-a)
    #print(denom)
    for i in range(0,k-1):
        theta_temp = copy.deepcopy(theta[:,i])
        x_i_temp = copy.deepcopy(x_i.T)
        denom = denom  + np.exp(np.matmul(theta_temp, x_i_temp) - a)

    prob = num/denom
    return prob

#x is x bar in column format
def cond_log_likeli(theta, x, y, k):
    log_likeli_sum = 0
    tempX = x.T


    for i in range(0,len(tempX)):

        prob_ = prob(y[i], tempX[i], theta, k)

        #max_prob = max(prob_list) #prob of one chosen
        log_likeli_sum = log_likeli_sum + np.log(prob_)
    
    log_likeli = (-1/len(x))*log_likeli_sum


    return log_likeli


#learn param
def logreg_fit(X, y, m, eta_start, eta_end, epsilon, max_epoch = 1000):
    #NEED TO ACCOUNT FOR MISSING CLASSES
    classes = find_classes(y) #should we assume 0 to be present
    global k
    k = len(classes)
    global loss_list
    loss_list = []
    
    X = X.T #apparently x is row vector

    #add one to end of each x
    ones = np.array([1 for i in range(0,len(X[0]))])
    x_bar = np.array(np.vstack((X, ones)))


    #get batches
    batches_x = []
    batches_y = []
    x_bar_temp = x_bar.T #row form
    for i in range(0,len(x_bar[0]), m):
        batches_x.append((x_bar_temp[i:i+m]))
        batches_y.append(y[i:i+m])

    batches_x = np.array(batches_x)
    batches_y = np.array(batches_y)

    ##randomly initialize theta
    theta = np.array(np.random.uniform(0,1, size= (len(X),len(classes)-1)))
    ones_theta = np.array([1 for i in range(0,len(classes)-1)]) #add bias
    theta = np.array(np.vstack((theta, ones_theta)))
    theta = theta.astype(np.float64)

    n=eta_start

    for i in range(0, max_epoch):
        theta_old = theta
        for m in range(0,len(batches_x)):
            theta = gradient_descent(batches_x[m], batches_y[m], theta, k, n)

        L_old = cond_log_likeli(theta_old, x_bar, y, k)
        L_new = cond_log_likeli(theta, x_bar, y, k)

        loss_list.append(L_new)
        if((L_old - L_new) < (epsilon * L_old)):
            n = n/10
        if(n< eta_end):
            break

    w = np.array(theta[:-1].T).reshape((k-1, len(theta)-1))
    bias = np.array(theta[-1])

    return [w, bias]

#predict prob for each class
def logreg_predict_prob(W, b, X):
    prediction_list = []
    theta = W.T
    theta = np.array(np.vstack((theta, b)))

    #[x;1]
    x_copy = copy.deepcopy(X)
    x_copy = x_copy.T #apparently x is row vector
    ones = np.array([1 for i in range(0,len(x_copy[0]))])
    x_copy = np.array(np.vstack((x_copy, ones)))
    #turns x to row vector
    tempX = x_copy.T

    for i in range(0, len(tempX)):
        prob_list = []
        #gets prob of each class getting predicted
        for j in range(0,k):
            prob_list.append(prob(j, tempX[i], theta, k))
        prediction_list.append(prob_list)
    
    prediction_list = np.array(prediction_list).reshape((len(tempX), k))
    return prediction_list

#predict data labels
def logreg_predict_class(W,b,X):
    predict_values = []
    theta = W.T
    theta = np.array(np.vstack((theta, b)))

    #[x;1]
    x_copy = copy.deepcopy(X)
    x_copy = x_copy.T #apparently x is row vector
    #print(x_copy)
    ones = np.array([1 for i in range(0,len(x_copy[0]))])
    #print("THIS IS X COPY")
    x_copy = np.array(np.vstack((x_copy, ones)))

    #turns it into a row vector
    tempX = x_copy.T


    for i in range(0, len(tempX)):
        prob_list = []
        #gets prob of getting classified for all classes
        for j in range(0,k):
            prob_list.append(prob(j, tempX[i], theta, k))
        
        #max val is predicted value
        max_prob_val = prob_list.index(max(prob_list))

        predict_values.append(max_prob_val)
    
    predict_values = np.array(predict_values)
    
    return predict_values

################################################################# PERFORMANCE METRICS ############################################################
#accuracy in percent
def accuracy(y_true, y_predicted):
    return accuracy_score(y_true, y_predicted)*100

def confusion_matrix_fun(y_true, y_predicted):
    return confusion_matrix(y_true, y_predicted), confusion_matrix(y_true, y_predicted, normalize='true')

#accuracy of diagonal of conf. matrix
def accuracy_diag_cm(cm):
    sum_values = []
    #cm = cm.T
    for i in range(0, len(cm)):
        sum_ =0
        diag = 0
        for j in range(0, len(cm[i])):
            sum_ = sum_ + cm[i][j]
            if(j==i):
                diag = diag + cm[i][j]
        #print(sum_)
        #print(diag)
        sum_values.append(diag/sum_)

    accur_diag = (np.sum(np.array(sum_values))/len(sum_values))*100
    return accur_diag

def prob_one(prob_):
    return prob_[:,1]

def average_precision(y_true, y1_score):
    return average_precision_score(y_true, y1_score)

def max_prob(prob_):
    return np.array([max(i) for i in prob_])


################################################################## GRAPHING #######################################################################
def plot_confusion(confusion_matrix, data_type):
    df_conf = pd.DataFrame(confusion_matrix, index = [i for i in range(0,len(confusion_matrix))],columns = [i for i in range(0,len(confusion_matrix))])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_conf, annot=True, cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix (rows sum to one) Given " + str(data_type) + " Data As Input" )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_precision_recall(y_true, y_pred):
    #print(precision_recall_curve(list(y_true), list(y_pred)))
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Precision Recall Curve")
    plt.show()

def plot_loss_epoch(loss_, set_name):
    #print("\n THIS WAS LOSS")
    #print(loss_)
    epoch = [i+1 for i in range(0,len(loss_))]
    plt.plot(epoch, loss_)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Loss Vs. Epochs Given " + set_name + " Data")
    plt.show()





################################################################ PRINTING #########################################################################
def print_weight_bias(weight, bias):
    print("\nWeight:\t" + str(weight))
    print("\nBias:\t" + str(bias))

def print_predict_prob(prob_, set_name):
    print("\nPredicted probabilities given " +str(set_name)+" data as input: ")
    for i in range(0,len(prob_)):
        print("\t"+ str(prob_[i]))

def print_predicted_out(out, set_name):
    print("\nPredicted output given " + str(set_name) + " data as input: ")
    print("\n[ ", end="")
    for i in range(0,len(out)):
        print(out[i], end=", ")
    print("]")

def print_accuracy(accuracy):
    print("\nAccuracy: " + str(accuracy)+"%")

def print_accuracy_cm_diag(accuracy):
    print("Accuracy along confusion matrix diagonal: " + str(accuracy) + "%")

def print_average_precision(avg_prec, class_name):
    print("Average Precision For "+ str(class_name)+ " Class: " + str(avg_prec))

############################################################ FUNCTION CALLS #################################################################

#get data
x_train, y_train = read_data("train.csv")
x_test, y_test = read_data("test.csv")

def normalized_train():
    x_train_norm, x_test_norm = normalize(x_train, x_test)
    [W, b] = logreg_fit(x_train_norm, y_train, 256, .01, .00001, .0001, 1000)
    plot_loss_epoch(loss_list, "Normalized")
    #print_weight_bias(W, b)

    #train data
    y_train_predict = logreg_predict_class(W, b, x_train_norm)
    y_train_prob = logreg_predict_prob(W,b,x_train_norm)
    accur_train = accuracy(y_train, y_train_predict)
    cm_train, cm_train_norm = confusion_matrix_fun(y_train, y_train_predict)
    accur_train_diag = accuracy_diag_cm(cm_train_norm)

    #graphing and printing
    print("\nUsing Training Data As Input: ")
    #print_predicted_out(y_train_predict, "training")
    #print_predict_prob(y_train_prob, "training")
    print_accuracy(accur_train)
    plot_confusion(cm_train_norm, "Training")
    print_accuracy_cm_diag(accur_train_diag)

    #test data
    y_test_predict = logreg_predict_class(W, b, x_test_norm)
    y_test_prob = logreg_predict_prob(W,b,x_test_norm)
    accur_test = accuracy(y_test, y_test_predict)
    cm_test, cm_test_norm = confusion_matrix_fun(y_test, y_test_predict)
    accur_test_diag = accuracy_diag_cm(cm_test_norm)

    #graphing and printing
    print("\nUsing Test Data As Input: ")
    #print_predicted_out(y_test_predict, "test")
    #print_predict_prob(y_test_prob, "test")
    print_accuracy(accur_test)
    plot_confusion(cm_test_norm, "Test")
    print_accuracy_cm_diag(accur_test_diag)

    #prob of pos class
    print("\nFor Only the Positive Test Case:")
    prob_test_pos = prob_one(y_test_prob)
    avg_prec_1 = average_precision(y_test, prob_test_pos)
    print_average_precision(avg_prec_1, "Positive")
    plot_precision_recall(y_test, prob_test_pos)

def normalized_train_dif_param():
    x_train_norm, x_test_norm = normalize(x_train, x_test)
    #base line
    [W, b] = logreg_fit(x_train_norm, y_train, 256, .01, .00001, .0001, 1000) #x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized")

    #eta_start
    #small
    [W_small_eta, b_small_eta] = logreg_fit(x_train_norm, y_train, 256, .00001, .00001, .0001, 1000) #x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Smaller Eta Start")
    #large
    [W_large_eta, b_large_eta] = logreg_fit(x_train_norm, y_train, 256, 10, .00001, .0001, 1000) #x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Larger Eta Start")
    #accuracy
    #small
    y_test_predict_eta_small = logreg_predict_class(W_small_eta, b_small_eta, x_test_norm)
    accur_test_eta_small = accuracy(y_test, y_test_predict_eta_small)
    print("Eta_start Small:")
    print_accuracy(accur_test_eta_small)
    #large
    y_test_predict_eta_large = logreg_predict_class(W_large_eta, b_large_eta, x_test_norm)
    accur_test_eta_large = accuracy(y_test, y_test_predict_eta_large)
    print("Eta_start Large:")
    print_accuracy(accur_test_eta_large)

    #eta_end
    #small
    [W_small_eta_end, b_small_eta_end] = logreg_fit(x_train_norm, y_train, 256, .01, .00000001, .0001, 1000) #x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Smaller Eta End")
    #large
    [W_large_eta_end, b_large_eta_end] = logreg_fit(x_train_norm, y_train, 256, .01, .01, .0001, 1000) #x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Larger Eta End")
    #accuracy
    #small
    y_test_predict_eta_small_end = logreg_predict_class(W_small_eta_end, b_small_eta_end, x_test_norm)
    accur_test_eta_small_end = accuracy(y_test, y_test_predict_eta_small_end)
    print("Eta_end Small:")
    print_accuracy(accur_test_eta_small_end)
    #large
    y_test_predict_eta_large_end = logreg_predict_class(W_large_eta_end, b_large_eta_end, x_test_norm)
    accur_test_eta_large_end = accuracy(y_test, y_test_predict_eta_large_end)
    print("Eta_end Large:")
    print_accuracy(accur_test_eta_large_end)

    #max_epoch
    #small
    [W_small_epoch, b_small_epoch] = logreg_fit(x_train_norm, y_train, 256, .01, .00001, .0001, 100)#x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Smaller Max Epoch")
    #large
    [W_large_epoch, b_large_epoch] = logreg_fit(x_train_norm, y_train, 256, .01, .00001, .0001, 10000)#x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Larger Max Epoch")
    #accuracy
    #small
    y_test_predict_small_epoch= logreg_predict_class(W_small_epoch, b_small_epoch, x_test_norm)
    accur_test_small_epoch = accuracy(y_test, y_test_predict_small_epoch)
    print("Max_epoch Small:")
    print_accuracy(accur_test_small_epoch)
    #large
    y_test_predict_large_epoch = logreg_predict_class(W_large_epoch, b_large_epoch, x_test_norm)
    accur_test_large_epoch = accuracy(y_test, y_test_predict_large_epoch)
    print("Max_epoch Large:")
    print_accuracy(accur_test_large_epoch)

    #m
    #small
    [W_small_m, b_small_m] = logreg_fit(x_train_norm, y_train, 100, .01, .00001, .0001, 1000)#x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Smaller M")
    #large
    [W_large_m, b_large_m] = logreg_fit(x_train_norm, y_train, 500, .01, .00001, .0001, 1000)#x, y, m, eta_start, eta_end, epsilon, max_epsilon
    plot_loss_epoch(loss_list, "Normalized with Larger M")
    #accuracy
    #small
    y_test_predict_small_m= logreg_predict_class(W_small_m, b_small_m, x_test_norm)
    accur_test_small_m = accuracy(y_test, y_test_predict_small_m)
    print("M Small:")
    print_accuracy(accur_test_small_m)
    #large
    y_test_predict_large_m = logreg_predict_class(W_large_m, b_large_m, x_test_norm)
    accur_test_large_m = accuracy(y_test, y_test_predict_large_m)
    print("M Large:")
    print_accuracy(accur_test_large_m)


    #MY OWN VALUES
def my_own_values():
    x_train_norm, x_test_norm = normalize(x_train, x_test)
    [W, b] =  logreg_fit(x_train_norm, y_train, 500, 10, .00000001, .0001, 10000)
    plot_loss_epoch(loss_list, "Normalized Own Values")
    y_test_predict= logreg_predict_class(W, b, x_test_norm)
    accur_test= accuracy(y_test, y_test_predict)
    print_accuracy(accur_test)

def non_normalized_train():
    [W, b] = logreg_fit(x_train, y_train, 256, .01, .00001, .0001, 1000) 
    plot_loss_epoch(loss_list, "NOT Normalized")
    print(W)
    print(b)

#normalized_train()
#non_normalized_train()
#my_own_values()
#normalized_train_dif_param()
##logreg_fit(temp_train, y_train, 256, .01, .00001, .0001, 1000)

