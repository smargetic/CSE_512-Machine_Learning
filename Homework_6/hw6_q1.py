#import torch
import sys
import os
print(sys.executable)
#FROM THIS DO "-m pip install _____library_____"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix

########################################################### SELF DEFINED NET ######################################################################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        #self.conv1 = nn.Conv2d(1, 3, 2)
        self.features1 = torch.nn.Sequential(
            nn.Conv2d(3, 3, 2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )

        self.conv2 = nn.Conv2d(3, 1, 2)

        self.features2 = torch.nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )
            

        self.classifier = nn.Sequential(
            nn.Linear(1*7*7, 25),
            nn.Linear(25, 10)
        )



    def forward(self, x):
        x = self.features1(x)
        x = self.conv2(x)
        x = self.features2(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x

############################################################## HELPER FUNCTIONS ################################################################
#CIFAR10 data
def get_data():
    #get data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

#for question 5
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    weights = None
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode


            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            print("si senior")
            if phase == 'train':
                scheduler.step()

            weights = model.state_dict()
    #not sure if this needs to be done
    model.load_state_dict(weights)
    return model
################################################################# EVALUATIONS ##################################################################
def accuracy(y_true, y_pred, class_labels):
    accuracy_all = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy_class = cm.diagonal()
    ##print("THIS IS ACCURACY CLASS")
    #print(accuracy_class)

    return accuracy_all, accuracy_class

def print_accuracy(accuracy_all, accuracy_class, class_labels):
    print("Accuracy for Entire Dataset:\t" + str(accuracy_all*100) + "%")
    print("Accuracy for Each Class:")
    for i in range(0,len(class_labels)):
        print(class_labels[i] +":\t" + str(accuracy_class[i]*100) + "%")

################################################################ QUESTION CALLS ##################################################################
#Question 1.1: Tensors
def question1():
    #a
    x = torch.rand((3,2),dtype=torch.float)
    print("\nX:")
    print(x)

    #b
    y = torch.ones_like(x)
    print("\nY:")
    print(y)
    print("Size: " + str(y.size()))

    #c
    #out = np.add(x, y)
    out = y.add(x)
    print("\nOut: ")
    print(out)
    y.add_(x)
    print("New Y: ")
    print(y)

    #d
    rand_np_array  = np.array(np.random.rand(3,2), dtype='float')
    print("\nRandom NP Array: ")
    print(rand_np_array)
    rand_torch_array = torch.from_numpy(rand_np_array)
    print("Tensor Version: ")
    print(rand_torch_array)
    convert_rand_np = rand_torch_array.numpy()
    print("Converted Back To NP Array:")
    print(convert_rand_np)

def question2():
    #a
    x = torch.rand((3,2), requires_grad=True)
    print("\nX:")
    print(x)

    #b
    y = x*10
    print("\nY - x*10:")
    print(y)

    y.add_(.1)
    print("Y - y+.1:")
    print(y)

    out = torch.max(y)
    print("Out: ")
    print(out)

    #c
    out.backward()
    gradient = x.grad
    print("\nD(out)/Dx:")
    print(gradient)

    #d
    with torch.no_grad():
            y = x*10
    print("\nY - x*10:")
    print(y)
 
    with torch.no_grad():
        y.add_(.1)
    print("Y - y+.1:")
    print(y)

    with torch.no_grad():
        out = torch.max(y)
    print("Out: ")
    print(out)

#create my own network
def question3():

    net = Net()
    print("Network:")
    print(net)
    params = list(net.parameters())
    print("Parameters:")
    print(params)
    number_param = sum(p.numel() for p in net.parameters())
    print("Number of Parameters:")
    print(number_param)

    #b
    rand_array = torch.randn(1, 3, 32, 32)
    output = net(rand_array)
    print("Network Output")
    print(output)
    print("Shape of Output:")
    print(output.shape)

    #c
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    output = net(rand_array)
    optimizer.zero_grad()

    rand_out = torch.randn(output.shape[0], output.shape[1])
    criterion = nn.MSELoss()
    loss = criterion(output, rand_out)


    net.zero_grad() 

    print('Conv2 Bias Grad Before Backpropogation')
    print(net.conv2.bias.grad)
    
    loss.backward()
    print('Conv2 Bias Grad After Backpropogation')
    print(net.conv2.bias.grad)

    optimizer.step()

def question4():
    #get data
    trainloader, testloader, classes= get_data()

    #initialize
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(3):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    #get predictions
    test_predictions = []
    test_truth = []
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = net(inputs)

        _, predicted = torch.max(outputs, 1)
        
        labels = labels.numpy()
        predicted = predicted.numpy()

        for j in range(0,len(labels)):
            test_predictions.append(predicted[j])
            test_truth.append(labels[j])

    #calculate accuracy
    accuracy_all, accuracy_class = accuracy(test_truth, test_predictions, classes)
    print_accuracy(accuracy_all, accuracy_class, classes)



def question5():
    #get data
    trainloader, testloader, classes = get_data()
    
    #change resnet to use number of classes we need
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    #run on cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    #NEED TO CHANGE
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,trainloader, device, 2)

    #get predictions
    test_predictions = []
    test_truth = []
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        
        labels = labels.numpy()
        predicted = predicted.numpy()

        for j in range(0,len(labels)):
            test_predictions.append(predicted[j])
            test_truth.append(labels[j])

    #calculate accuracy
    accuracy_all, accuracy_class = accuracy(test_truth, test_predictions, classes)
    print_accuracy(accuracy_all, accuracy_class, classes)
    
    #part b
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(classes))

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,trainloader, device, 2)

    #get predictions
    test_predictions = []
    test_truth = []
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)
        
        labels = labels.numpy()
        predicted = predicted.numpy()

        for j in range(0,len(labels)):
            test_predictions.append(predicted[j])
            test_truth.append(labels[j])

    print("\n")
    #calculate accuracy
    accuracy_all, accuracy_class = accuracy(test_truth, test_predictions, classes)
    print_accuracy(accuracy_all, accuracy_class, classes)





#question1()
#question2()
#question3()
#question4()
question5()
