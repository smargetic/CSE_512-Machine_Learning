import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

sum_square_errors = []

def read_data(fileName):
    data = []
    label = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = 1
            else:
                label.append(row[0])
                data.append(row[1::])
    
    data = np.array(data).astype(int)
    label = np.array(label).astype(int)

    data = np.true_divide(data, 255.0) #DOUBLE CHECK THIS
    return data, label

def assignment(X, centroids):
    min_dist_cluster = []
    global sum_square_errors
    total_dist = 0
    for i in range(0,len(X)):
        #get min distance b/w center and point
        distance_list = []
        for j in range(0,len(centroids)): #for all centers
            
            temp_dist = 0
            for m in range(0,len(centroids[j])):
                temp_dist = ((centroids[j][m] - X[i][m])**2) + temp_dist

            distance_list.append(temp_dist)
            #distance_list.append((np.linalg.norm(centroids[j]-[X[i]]))**2) #WAS OFF FOR EXTREMELY SMALL DIGITS

        #find min ==> where pt gets assigned
        min_loc = distance_list.index(min(distance_list))
        min_dist_cluster.append(min_loc)

        #smallest dist = where pt was assigned ==> use for total_dist
        total_dist = total_dist + distance_list[min_loc]
    
    sum_square_errors.append(total_dist)
    min_dist_cluster = np.array(min_dist_cluster)

    return min_dist_cluster

def k_means(X, k):
    #global sum_square_errors
    centroids = []
    centroids_old = []
    cluster_groups = {}
    #get random data points to initialize cluster centers
    for i in range(0,k):
        centroids.append(X[random.randint(0, len(X)-1)])
        centroids_old.append(X[random.randint(0, len(X)-1)]-1)
    
    #np array
    centroids = np.array(centroids)
    centroids_old = np.array(centroids_old)

    while(not (centroids == centroids_old).all()):
        centroids_old = centroids

        #init 
        for i in range(0,k):
            cluster_groups[i] = []

        #assign points to cluster
        min_dist_cluster = assignment(X,centroids)
        #sort by clusters
        for i in range(0,len(min_dist_cluster)):
            cluster_groups[min_dist_cluster[i]].append(i)

        #recalculate center for each cluster
        for i in cluster_groups.keys(): #go through each cluster
            #get sum in each dimension
            sum_dim = [0]*len(X[0])
            for j in range(0,len(cluster_groups[i])):
                for m in range(0,len(X[cluster_groups[i][j]])):
                    sum_dim[m] = sum_dim[m] + X[cluster_groups[i][j]][m]

            #new centroid is average of all data pts in cluster
            sum_dim = np.array(sum_dim)
            new_centroid = np.true_divide(sum_dim,len(cluster_groups[i])) #average value
            centroids[i] = new_centroid
            
            centroids = np.array(centroids)
            #centroids_old = np.array(centroids_old)
            #print((centroids == centroids_old).all())

    centroids = np.array(centroids).reshape((len(centroids),len(centroids[0]))) #DOUBLE CHECK THIS
    return centroids

############################################################### GRAPHING #############################################################################
def plot_(y_data, name):
    x_data = []
    for i in range(0,len(y_data)):
        x_data.append(i+1)
    plt.plot(x_data,y_data)
    plt.ylabel(name)
    plt.xlabel("Iterations")
    plt.title("Iterations Versus " + name)
    plt.show()

############################################################## PRINTING ##############################################################################
def print_sum_square(sum_square):
    print("\nSum Of Square at Each Iteration: " + str(sum_square))
    print("\nFinal Value of Sum of Squared Errors: " + str(sum_square[-1]))

############################################################## FUNCTION CALLS #########################################################################

#question 1.2
def question_12(data_train, k):
    #data_train, label_train = read_data("mnist_train_hw5.csv")
    centroids = k_means(data_train, k)
    plot_(sum_square_errors, "Sum Of Squared Errors")
    print_sum_square(sum_square_errors)
    return centroids

#question 1.3
def question_13(data_test, centriods, k):
    min_dist_cluster = assignment(data_test,centriods)
    count = {}
    #init
    for i in range(0,len(centriods)):
        count[i] = 0

    #count how many belong to each cluster
    for i in range(0,len(min_dist_cluster)):
        count[min_dist_cluster[i]] = count[min_dist_cluster[i]] + 1


    fig = plt.figure()
    rows = 2
    cols = 5
    if(k==8):
        cols = 4
    if(k==12):
        cols = 6
    for i in range(0,len(centriods)):
        temp = centriods[i]*255.0
        #NOT SURE IF SHOULD BE UNIT *
        #reshape image
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.reshape(temp, (28,28)))
        plt.axis("off")
        plt.title("Count: " + str(count[i]))
    plt.tight_layout()
    plt.show()

def question_14(data_train, data_test, k_extra):
    for i in range(0,len(k_extra)):
        centroids = question_12(data_train, k_extra[i])
        question_13(data_test, centroids, k_extra[i])    

#question_1()
data_train, label_train = read_data("mnist_train_hw5.csv")
data_test, label_test = read_data("mnist_test_hw5.csv")
centroids = question_12(data_train, 10)
question_13(data_test, centroids, 10)

#question 4
k_extra = [8, 12]
question_14(data_train,data_test,k_extra)
