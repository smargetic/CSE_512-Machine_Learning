import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from data_preprocessing import read_data, change_categorical, read_time_series

import sys
from sklearn import linear_model
import copy

attributeNames = ["Age", "Gender"]

#question 2
def get_mean_and_variance(X, y):
    mu0 = []
    var0 = []
    mu1 = []
    var1 = []
    num0 = 0
    num1 = 0

    #initialize 
    for i in range(0,len(X[0])):
        mu0.append(0)
        var0.append(0)
        mu1.append(0)
        var1.append(0)

    #calculate sum for respective outcomes
    for i in range(0,len(X)):
        if(y[i] == 0):
            num0 = num0 + 1
            for j in range(0,len(X[i])):
                mu0[j] = mu0[j] + X[i][j]
        elif(y[i] == 1):
            num1 = num1 + 1
            for j in range(0,len(X[i])):
                mu1[j] = mu1[j] + X[i][j]

    # mean
    if(num0 > 0):
        for i in range(0, len(mu0)):
            mu0[i] = mu0[i]/num0

    if(num1 > 0):
        for i in range(0, len(mu1)):
            mu1[i] = mu1[i]/num1

    # calculate the square difference b/w data and mean
    for i in range(0,len(X)):
        if(y[i]==0):
            for j in range(0,len(X[i])):
                var0[j] = var0[j] + (X[i][j] - mu0[j])**2
        else:
            for j in range(0,len(X[i])):
                var1[j] = var1[j] + (X[i][j] - mu1[j])**2

    # divide by the number of data points
    if(num0 > 0):
        var0 = np.divide(var0, (num0-1))

    if(num1 > 1):
        var1 = np.divide(var1, (num1-1))

    # convert list into numpy array
    mu0 = np.array(mu0)
    var0 = np.array(var0)
    mu1 = np.array(mu1)
    var1 = np.array(var1) 

    return [mu0, var0, mu1, var1]

#question 3 --> learn linear reg param
def learn_reg_params(x, y):
    input_ = []
    output = []
    for i in range(6,len(x)-1):
        cases = []
        deaths = []
        for j in range(i, i-7, -1):
            cases.append(x[j])
            deaths.append(y[j])

        joined_list = cases + deaths
        input_.append(joined_list)
        output.append(y[i+1])

    input_ = np.array(input_)
    output = np.array(output)

    reg = linear_model.LinearRegression()
    reg.fit(input_, output)

    return reg.coef_, reg.intercept_, reg, input_

#gaussian model for error parameters
def gaussian_errors(reg, input_, deaths):
    predicted_output = reg.predict(input_)
    deaths_ = copy.deepcopy(deaths)
    deaths_ = deaths_[7:]
    error = []
    error = deaths_ - predicted_output
    error = np.array(error)

    mean_error = np.mean(error)
    var_error = np.var(error, ddof=1)
    return mean_error, var_error, error


######################################################## GRAPHING ###################################################################333
#question 2
def generate_gaussian(mu0, var0, mu1, var1, output_changeList, gender_changeList):
    for i in range(0, len(mu0)):
        sigma0 = math.sqrt(var0[i])
        sigma1 = math.sqrt(var1[i])

        #find x range
        minValue = 0
        maxValue = 0
        if((mu0[i]-3*sigma0) < (mu1[i]-3*sigma1)):
            minValue = mu0[i]-3*sigma0
        else:
            minValue = mu1[i]-3*sigma1

        if((mu0[i] + 3*sigma0) > (mu1[i] + 3*sigma1)):
            maxValue = mu0[i] + 3*sigma0
        else:
            maxValue = mu1[i] + 3*sigma1

        x0 = np.linspace(minValue, maxValue, 500)
        x1 = np.linspace(minValue, maxValue, 500) 

        plt.plot(x0, norm.pdf(x0, mu0[i], sigma0), 'b', label = output_changeList[0][0] )
        plt.plot(x1, norm.pdf(x1, mu1[i], sigma1), 'r', label = output_changeList[0][1])
        plt.xlabel(attributeNames[i])
        plt.ylabel("Probability Density")
        title = "Gaussian Distribution for: " + str(attributeNames[i])
        if(attributeNames[i]=="Gender"):
            plt.xticks([0,1], gender_changeList[0])
        plt.title(title)
        plt.legend()
        plt.show()

#question 3
def plot_linearReg(confirmed, deaths, reg, input_):
    predicted_output = reg.predict(input_)
    confirmed_ = copy.deepcopy(confirmed)
    deaths_ = copy.deepcopy(deaths)
    confirmed_ = confirmed_[7:]
    deaths_ = deaths_[7:]

    plt.scatter(confirmed_, deaths_,c='b', label = "True Data")
    plt.scatter(confirmed_, predicted_output, c='r', label = "Predicted Data")
    plt.xlabel("Confirmed Cases")
    plt.locator_params(axis='x', nbins=20)
    plt.ylabel("Deaths")
    plt.title("Linear Regression Predicted Values and Orginal Values")
    plt.legend()
    plt.show()

#question 3
def gaussian_and_hist(mean, var, error):
    #gaussian
    minValue = min(error)
    maxValue = max(error)
    x = np.linspace(minValue, maxValue, 500)

    plt.plot(x, norm.pdf(x, mean, math.sqrt(var)), 'b')

    #histogram
    plt.hist(error, bins=70, alpha=.5, density=True) #normalized
    plt.xlabel("Error")
    plt.ylabel("Probability Density")
    #plot
    plt.title("Error for yt − y^t")
    plt.show()


####################################### PRINTING ###################################################################################

def print_mean_and_var(mu0, var0, mu1, var1):
    print("mu0: " + str(mu0))
    print("var0: " + str(var0))
    print("mu1: " + str(mu1))
    print("var1: " + str(var1))

def print_w_and_int(w, intercept):
    print("W: " + str(w))
    print("Y-intercept: " + str(intercept))

def print_gaussian_errors(mean, variance):
    print("\nLinear Regression Model Errors Represented as a Gaussian:")
    print("\tMean: "+ str(mean))
    print("\tVariance: " + str(variance))

############################################################# SPECIFIC PROBLEM SOLUTION CALLS #####################################################

#using built functions to answer question 2 in specific
def question2_functions():
    metadata, metadata_output = read_data('covid19_metadata.csv')
    fixed_metadata, metadata_changeList = change_categorical(metadata)
    fixed_metadata_output, output_changeList = change_categorical(metadata_output)

    [mu0, var0, mu1, var1] = get_mean_and_variance(fixed_metadata,fixed_metadata_output)
    print_mean_and_var(mu0, var0, mu1, var1)
    generate_gaussian(mu0, var0, mu1, var1, output_changeList, metadata_changeList)

#using built functions to answer question 3 in specific
def question3_functions():
    dates, confirmed, deaths= read_time_series('covid19_time_series.csv')
    w, intercept, reg, input_ = learn_reg_params(confirmed, deaths)
    print_w_and_int(w, intercept)

    plot_linearReg(confirmed, deaths, reg, input_)
    mean, var, error = gaussian_errors(reg, input_, deaths)
    print_gaussian_errors(mean, var)
    gaussian_and_hist(mean, var, error)


#question2_functions()
#question3_functions()


