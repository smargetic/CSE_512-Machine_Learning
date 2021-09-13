import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import GridSearchCV
import sys

############################################################## PREPROCESSING ##########################################################################

def read_data(fileName):
    data = np.genfromtxt(fileName, dtype=np.object, delimiter=',')
    #seperate output
    output = [x[len(x)-1] for x in data]
    output = np.array(output)
    #remove output from orig data
    data = [x[0:len(x)-1] for x in data]
    data = np.array(data)

    return data, output

def convert_categorial_to_float(train_data, train_output, test_data, test_output):
    conversion_output = [list(set(train_output))]

    train_data = train_data.T
    test_data = test_data.T

    categorical_list_train = []
    categorical_list_test = []
    categorical_list_index = []
    #get indeces of categorical attributes seperately
    for i in range(0,len(train_data)):
        try:
            float(train_data[i][0]) 
        except ValueError: #categorical
            categorical_list_train.append(train_data[i])
            categorical_list_test.append(test_data[i])
            categorical_list_index.append(i)
    
    #attach output
    categorical_list_train.append(train_output)
    categorical_list_test.append(test_output)

    #return to [num samples, features] shape
    categorical_list_train = np.array(categorical_list_train).T
    categorical_list_test = np.array(categorical_list_test).T

    #train and fit data
    enc = OrdinalEncoder()
    new_train_cat = enc.fit_transform(categorical_list_train)
    new_test_cat = enc.transform(categorical_list_test)
    

    new_train_cat = np.array(new_train_cat).T
    new_test_cat = np.array(new_test_cat).T

    #convert original values 
    train_data_new = [new_train_cat[categorical_list_index.index(i)] if i in categorical_list_index else train_data[i] for i in range(0,len(train_data))]
    test_data_new = [new_test_cat[categorical_list_index.index(i)] if i in categorical_list_index else test_data[i] for i in range(0,len(test_data))]

    train_data_new = np.array(train_data_new).T
    test_data_new = np.array(test_data_new).T

    #output values
    train_output_new = new_train_cat[-1]
    test_output_new = new_test_cat[-1]

    conversion_output.append(list(set(train_output_new)))

    return train_data_new, train_output_new, test_data_new, test_output_new, conversion_output

#convert to float ==> did for all data values as to save space in function call section
def convert_to_float(train_data, train_output, test_data, test_output):
    train_data = train_data.astype(np.float32)
    train_output = train_output.astype(np.float32)
    test_data = test_data.astype(np.float32)
    test_output = test_output.astype(np.float32)

    return train_data, train_output, test_data, test_output

######################################################## XGBoost ###################################################################################
def xgboost_implementation(train_data, train_output, test_data):
    #training
    model = XGBClassifier()
    model.fit(train_data, train_output)

    #prediction
    train_output_pred = model.predict(train_data)
    test_output_pred = model.predict(test_data)

    return train_output_pred, test_output_pred

#def xgboost_implementation2(train_data, train_output, test_data):
#    xgb.DMatrix(train_data, label=train_output)


def k_fold(train_data,train_output, test_data, test_output, conversion_output, number_splits=10):
    kf = KFold(n_splits=number_splits)
    kf.get_n_splits(train_data)

    max_values = {'max_depth':-0, 'min_child_weight':0, 'gamma': 0,
                    'subsample': 0, 'colsample_bytree':0, 'reg_alpha':0,
                    "accuracy": -sys.maxsize}

    model = XGBClassifier()
    #SEE IF THIS GOES THROUGH
    model.fit(train_data, train_output) 

    for train_index, test_index in kf.split(train_data):
        # split the data based on these indecies
        axis = 0 # NEED TO MAKE SURE THIS IS CORRECT
        
        split_train_data = np.take(train_data, train_index, axis)
        split_train_output = np.take(train_output, train_index, axis)

        split_test_data = np.take(train_data, test_index, axis) 
        split_test_output = np.take(train_output, test_index, axis)
        
        #set parameters to a default set of values
        #xgb1 = XGBClassifier(
        #    learning_rate =0.1,
        #    n_estimators=1000,
        #    max_depth=5,
        #    min_child_weight=1,
        #    gamma=0,
        #    subsample=0.8,
        #    colsample_bytree=0.8,
        #    objective= 'binary:logistic',
        #    nthread=4,
        #    scale_pos_weight=1,
        #    seed=27)
        
        #gsearch1.fit(train[predictors],train[target])
        #see how well the model behaves with these
        #accuracy_value = modelfit(xgb1, split_train_data, split_train_output,useTrainCV=True, cv_folds=5, early_stopping_rounds=50)

        #tune max_depth and min_child_weight
        param_test1 = {
            'max_depth':range(3,10,2),
            'min_child_weight':range(1,10,2)
        }
        gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27, verbosity = 0,use_label_encoder =False), 
        param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4, cv=5, return_train_score = True)
        gsearch1.fit(split_train_data,split_train_output)

        max_depth_b = gsearch1.best_params_["max_depth"]
        min_child_w_b = gsearch1.best_params_["min_child_weight"]

        #if(min_child_w_b!=1):
        param_test2 = {
            'max_depth': [max_depth_b-1, max_depth_b, max_depth_b+1],
            'min_child_weight': [min_child_w_b-1,min_child_w_b, min_child_w_b+1] #NOT SURE IF I SHOULD DO SOMETHING SEPERATE FOR 0
    
            }

        #else:
        #    param_test2 = {
        #        'max_depth': [max_depth_b-1, max_depth_b, max_depth_b+1],
        #        'min_child_weight': [min_child_w_b, min_child_w_b+1] #NOT SURE IF I SHOULD DO SOMETHING SEPERATE FOR 0

        #    }
        gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27, verbosity = 0, 
        use_label_encoder =False), 
        param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=4, cv=5, return_train_score = True)
        gsearch2.fit(split_train_data,split_train_output)
        
        #tune gamma
        param_test3 = {
            'gamma':[i/10.0 for i in range(0,5)]
        }
        gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=gsearch2.best_params_["max_depth"],
        min_child_weight=gsearch2.best_params_["min_child_weight"], gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0, use_label_encoder =False), param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=4,cv=5, 
        return_train_score = True)
        gsearch3.fit(split_train_data,split_train_output)

        #tune subsample and colsample_bytree
        param_test4 = {
        'subsample':[i/10.0 for i in range(2,10)],
        'colsample_bytree':[i/10.0 for i in range(2,10)]
        }

        gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=gsearch2.best_params_["max_depth"],
        min_child_weight=gsearch2.best_params_["min_child_weight"], gamma=gsearch3.best_params_["gamma"], subsample=0.8, colsample_bytree=0.8,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0, use_label_encoder =False), 
        param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs=4, cv=5, return_train_score = True)
        
        gsearch4.fit(split_train_data,split_train_output)
        
        #narrow into best value
        param_test5 = {
        'subsample':[i/100.0 for i in range(int((gsearch4.best_params_["subsample"]*100)-8),int((gsearch4.best_params_["subsample"]*100)+8),2)],
        'colsample_bytree':[i/100.0 for i in range(int((gsearch4.best_params_['colsample_bytree']*100)-8),int((gsearch4.best_params_['colsample_bytree']*100)+8),2)]
        }

        gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=gsearch2.best_params_["max_depth"],
        min_child_weight=gsearch2.best_params_["min_child_weight"], gamma=gsearch3.best_params_["gamma"], subsample=0.8, colsample_bytree=0.8,
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0, use_label_encoder =False), 
        param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=4,cv=5, return_train_score = True)
        
        gsearch5.fit(split_train_data,split_train_output)

        #tune regularization param
        param_test6 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
        }
        gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=gsearch2.best_params_["max_depth"],
        min_child_weight=gsearch2.best_params_["min_child_weight"], gamma=gsearch3.best_params_["gamma"], subsample=gsearch5.best_params_["subsample"], 
        colsample_bytree=gsearch5.best_params_["colsample_bytree"], objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0,
        use_label_encoder =False), 
        param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs=4, cv=5, return_train_score = True)

        gsearch6.fit(split_train_data,split_train_output)
        #narrow down on value
        min_value = gsearch6.best_params_["reg_alpha"] - (gsearch6.best_params_["reg_alpha"]*.9)
        max_value = gsearch6.best_params_["reg_alpha"] + (gsearch6.best_params_["reg_alpha"]*.9)
        #step_value = (max_value - min_value)/6
        param_test7 = {
            'reg_alpha': list(np.linspace(min_value,max_value,30))
        #'reg_alpha':[i for i in range(min_value, max_value, step_value)]
        }
        gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=gsearch2.best_params_["max_depth"],
        min_child_weight=gsearch2.best_params_["min_child_weight"], gamma=gsearch3.best_params_["gamma"], subsample=gsearch5.best_params_["subsample"], 
        colsample_bytree=gsearch5.best_params_["colsample_bytree"], objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0,
        use_label_encoder =False), 
        param_grid = param_test7, scoring='neg_mean_squared_error',n_jobs=4, cv=5, return_train_score = True)

        gsearch7.fit(split_train_data,split_train_output)

        # SHOULD DO FURTHER############################################
        #reduce learning rate ==> DOUBLE CHECK THIS STEP
        xgb_final =  XGBClassifier( learning_rate =0.01, n_estimators=177, max_depth=gsearch2.best_params_["max_depth"],
        min_child_weight=gsearch2.best_params_["min_child_weight"], gamma=gsearch3.best_params_["gamma"], subsample=gsearch5.best_params_["subsample"], 
        colsample_bytree=gsearch5.best_params_["colsample_bytree"], 
        reg_alpha  = gsearch7.best_params_["reg_alpha"],
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0, use_label_encoder =False)

        xgb_final.fit(split_train_data, split_train_output)
        predict_output = xgb_final.predict(split_test_data)
        accuracy_value = accuracy_score(split_test_output, predict_output)*100


        #if accuracy higher than max values, update max values
        if(accuracy_value>max_values["accuracy"]):
            max_values['max_depth'] = gsearch2.best_params_["max_depth"]
            max_values['min_child_weight'] = gsearch2.best_params_["min_child_weight"]
            max_values['gamma'] =  gsearch3.best_params_["gamma"]
            max_values['subsample'] = subsample=gsearch5.best_params_["subsample"]
            max_values['colsample_bytree'] = gsearch5.best_params_["colsample_bytree"]
            max_values['reg_alpha'] =  .01
        #print("at end")

    #with best results, report accuraccy and confusion matrix
    xgb_test =  XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=max_values['max_depth'],
        min_child_weight=max_values['min_child_weight'], gamma=max_values['gamma'], subsample=max_values['subsample'], 
        colsample_bytree=max_values['colsample_bytree'], 
        reg_alpha  = max_values['reg_alpha'],
        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27, verbosity = 0, use_label_encoder =False)  

    xgb_test.fit(train_data, train_output)
    predicted_test = xgb_test.predict(test_data)  

    print_accuracy(test_output, predicted_test, "Test") #SHOULD CHANGE TITLE FOR OPT VALUES
    plot_confusion(test_output, predicted_test, "Test", conversion_output)



##################################################### PERFORMANCE METRICS ###########################################################################
#def confusion_matrix_fun(y_true, y_predicted):
#    return confusion_matrix(y_true, y_predicted), confusion_matrix(y_true, y_predicted, normalize='true')

##################################################### GRAPHING ####################################################################################
def plot_confusion(y_true, y_pred, data_type, conversion_output):
    #calculate confusion matrix
    cm, cm_norm = confusion_matrix(y_true, y_pred), confusion_matrix(y_true, y_pred, normalize='true')

    df_conf = pd.DataFrame(cm_norm, index = [i for i in conversion_output[0]],columns = [i for i in conversion_output[0]])

    plt.figure(figsize = (10,7))
    sn.heatmap(df_conf, annot=True, cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix (rows sum to one) Given " + str(data_type) + " Data As Input" )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

######################################################### PRINTING ##################################################################################
def print_accuracy(y_true, y_pred, train_or_test):
    accuracy_value = accuracy_score(y_true, y_pred)*100
    print("Accuracy for " + train_or_test + " Data:\t" + str(accuracy_value) + "%")


##################################################### FUNCTION CALLS ###############################################################################
def question2_1():
    train_data, train_output = read_data("adult.data")
    test_data, test_output = read_data("adult.test")
    train_data, train_output, test_data, test_output, conversion_output = convert_categorial_to_float(train_data, train_output, test_data, test_output)
    train_data, train_output, test_data, test_output = convert_to_float(train_data, train_output, test_data, test_output)
    train_predict, test_predict = xgboost_implementation(train_data, train_output, test_data)
    #train set
    print_accuracy(train_output, train_predict, "Train")
    plot_confusion(train_output, train_predict, "Training", conversion_output)

    #test set 
    print_accuracy(test_output, test_predict, "Test")
    plot_confusion(test_output, test_predict, "Test", conversion_output)

def question2_2():
    train_data, train_output = read_data("adult.data")
    test_data, test_output = read_data("adult.test")
    train_data, train_output, test_data, test_output, conversion_output = convert_categorial_to_float(train_data, train_output, test_data, test_output)
    train_data, train_output, test_data, test_output = convert_to_float(train_data, train_output, test_data, test_output)
    k_fold(train_data, train_output, test_data, test_output, conversion_output)
    
#has both part 1 and part 2 as to accurately represent outcome of tuning hyperparam    
def question_2():
    train_data, train_output = read_data("adult.data")
    test_data, test_output = read_data("adult.test")
    train_data, train_output, test_data, test_output, conversion_output = convert_categorial_to_float(train_data, train_output, test_data, test_output)
    train_data, train_output, test_data, test_output = convert_to_float(train_data, train_output, test_data, test_output)
    
    #part 1
    train_predict, test_predict = xgboost_implementation(train_data, train_output, test_data)
    #train set
    print_accuracy(train_output, train_predict, "Train")
    plot_confusion(train_output, train_predict, "Training", conversion_output)

    #test set 
    print_accuracy(test_output, test_predict, "Test")
    plot_confusion(test_output, test_predict, "Test", conversion_output)

    #part 2
    k_fold(train_data, train_output, test_data, test_output, conversion_output)



#question2_1()
#question2_2()
question_2()