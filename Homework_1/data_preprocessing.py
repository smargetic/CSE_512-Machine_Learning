import csv
import numpy as np
import datetime

#read data for question 2
def read_data(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0 # should change slightly

        data = []
        output = []
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
            else:
                data.append(row[:-1])
                output.append(row[-1])

        data = np.array(data)
        output = np.array(output)
        return data, output

# changes categorical values to numeric
# also changes string to int
def change_categorical(data):
    changeList = []

    # i store all possible categorical values in a seperate list
    for i in range(0, len(data)):
        count = 0
        for j in range(0, len(data[i])):
            try:
                data[i][j] = int(data[i][j])
            except ValueError:
                if(i == 0):
                    changeList.append([data[i][j]])
                else:
                    if(data[i][j] not in changeList[count]):
                        changeList[count].append(data[i][j])
                    count = count + 1

    # convert values
    for i in range(0, len(data)):
        count = 0
        
        if(len(data[i])!=1):
            for j in range(0, len(data[i])):
                try:
                    data[i][j] = int(data[i][j])
                except ValueError:
                    data[i][j] = changeList[count].index(data[i][j])
                    count = count + 1
        else:
            try:
                data[i] = int(data[i])
            except ValueError:
                data[i] = changeList[0].index(data[i])
    

    data = np.array(data, dtype=int) #make sure values are ints
    return data, changeList

#read data for question 3
def read_time_series(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0 # should change slightly

        dates = []
        confirmed = []
        deaths = []

        
        for row in csv_reader:
            if line_count == 0:
                dates.append(row[1:])
            elif(line_count == 1):
                confirmed.append(row[1:])
            elif(line_count == 2):
                deaths.append(row[1:])
            line_count = line_count + 1

        confirmed = list(map(int, confirmed[0]))
        deaths = list(map(int, deaths[0]))

        dates = np.array(dates[0])
        confirmed = np.array(confirmed)
        deaths = np.array(deaths)
        return dates, confirmed, deaths