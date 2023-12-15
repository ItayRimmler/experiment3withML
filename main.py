import sys
import random as r
import torch as t
import torch.nn as nn
import numpy as np
# import sklearn as s
import matplotlib.pyplot as plt
from exceptions import *

# Let's practice simple use of torch, we shall create an exp dataset, with random noise, then see how can we use ML:

# CONSTANTS:

try:

    DATA_SET_SIZE = 30
    BATCH_PERCENT = 0.35
    EPOCHS = 500
    LEARNING_RATE = 0.01
    FEATURES_NUMBER = 1
    NOISE = 0.02
    ACCURACY_COEFF = 0.3

    DATA_SET_SIZE = round(DATA_SET_SIZE)
    EPOCHS = round(EPOCHS)
    BATCH_SIZE = round(DATA_SET_SIZE * BATCH_PERCENT)
    FEATURES_NUMBER = round(FEATURES_NUMBER)

    exceptionz = []
    exception1 = DATA_SET_SIZE >= 1
    exceptionz.append(exception1)
    exception2 = 0 < BATCH_PERCENT <= 1
    exceptionz.append(exception2)
    exception3 = EPOCHS >= 1
    exceptionz.append(exception3)
    exception4 = BATCH_SIZE >= 1
    exceptionz.append(exception4)
    exception5 = 0 < LEARNING_RATE <= 1
    exceptionz.append(exception5)
    exception6 = FEATURES_NUMBER >= 1
    exceptionz.append(exception6)
    exception7 = 0 <= NOISE <= 1
    exceptionz.append(exception7)
    exception8 = 0 < ACCURACY_COEFF < 1
    exceptionz.append(exception8)
    exceptionz = np.array(exceptionz)

    if not exceptionz.any():
        raise tooSmall(name='ALL')
    if (exceptionz == False).sum() >= 2:
        raise tooSmall(name='ANY')
    if not exception1:
        raise tooSmall(name='DATA_SET_SIZE', b=DATA_SET_SIZE)
    if not exception2:
        raise tooSmall(name='BATCH_PERCENT', b=BATCH_PERCENT)
    if not exception3:
        raise tooSmall(name='EPOCHS', b=EPOCHS)
    if not exception4:
        raise tooSmall(name='BATCH_SIZE', b=BATCH_SIZE)
    if not exception5:
        raise tooSmall(name='LEARNING_RATE', b=LEARNING_RATE)
    if not exception6:
        raise tooSmall(name='FEATURES_NUMBER', b=FEATURES_NUMBER)
    if not exception7:
        raise tooSmall(name='NOISE', b=NOISE)
    if not exception8:
        raise tooSmall(name='ACCURACY_COEFF', a=0.9999999999, b=ACCURACY_COEFF)

except tooSmall as e:
    if e.name == 'ALL':
        DATA_SET_SIZE = e.b
        BATCH_PERCENT = e.b
        EPOCHS = e.b
        BATCH_SIZE = e.b
        LEARNING_RATE = e.b
        NOISE = e.b
        ACCURACY_COEFF = e.b - 0.0000000001
    if e.name == 'DATA_SET_SIZE':
        DATA_SET_SIZE = e.b
    if e.name == 'BATCH_PERCENT':
        BATCH_PERCENT = e.b
    if e.name == 'EPOCHS':
        EPOCHS = e.b
    if e.name == 'BATCH_SIZE':
        BATCH_SIZE = e.b
    if e.name == 'LEARNING_RATE':
        LEARNING_RATE = e.b
    if e.name == 'NOISE':
        NOISE = e.b
    if e.name == 'ACCURACY_COEFF':
        ACCURACY_COEFF = e.b - 0.0000000001
    else:
        sys.exit()
finally:


    # Building the dataset:

    xBase = np.array(range(0, DATA_SET_SIZE))
    yBase = np.exp(xBase)

    noise = np.random.uniform(-NOISE, NOISE, size=DATA_SET_SIZE)

    xTrain = xBase
    yTrain = np.round(yBase + noise * yBase, decimals=10)

    xTest = np.array([DATA_SET_SIZE])
    yTest = np.round(np.array([np.exp(xTest)]), decimals=10)

    # Preprocessing the dataset:

    xTrain = t.from_numpy(xTrain.astype(np.float32))
    xTrain = xTrain.view(-1, FEATURES_NUMBER)
    yTrain = t.from_numpy(yTrain.astype(np.float32))
    yTrain = yTrain.view(-1, FEATURES_NUMBER)
    xTest = t.from_numpy(xTest.astype(np.float32))
    xTest = xTest.view(-1, FEATURES_NUMBER)
    xTest.requires_grad_(True)
    yTest = t.from_numpy(yTest.astype(np.float32))
    yTest = yTest.view(-1, FEATURES_NUMBER)

    # Building the model:


    def batch(x, y, size):
        try:

            size = round(size)

            if size > DATA_SET_SIZE or size < 1:
                raise tooSmall(DATA_SET_SIZE, size)

        except tooSmall as e:
            size = e.b

        ind = sorted(r.sample(range(0, DATA_SET_SIZE), size))
        x = x[ind]
        y = y[ind]
        return x, y


    class ExpRegModel(nn.Module):

        def __init__(self, inputsSize, outputSize):
            super().__init__()
            self.lay1 = nn.Linear(in_features=inputsSize, out_features=round(DATA_SET_SIZE * 6.4))
            self.lay2 = nn.Linear(in_features=round(DATA_SET_SIZE * 6.4), out_features=outputSize)

        def forward(self, inputs):
            inputs = t.sigmoid(self.lay1(inputs))
            outputs = self.lay2(inputs)
            return outputs


    # Defining the model:
    model = ExpRegModel(FEATURES_NUMBER, FEATURES_NUMBER)
    criterion = nn.HuberLoss()
    optimizer = t.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    # Training:
    for epoch in range(EPOCHS):

        xTrain.requires_grad_(True)

        # Preparing the batch:
        inputs, labels = batch(xTrain, np.log(yTrain), BATCH_SIZE)

        # Setting the optimizer:
        optimizer.zero_grad()

        # Inserting inputs:
        outputs = model(inputs)

        # Calculating loss:
        loss = criterion(outputs, labels)

        # Fitting:
        loss.backward()
        optimizer.step()

        # Calculating accuracy (I'm still not sure how can I do it better):

        acc1 = t.gt(outputs, (1 - ACCURACY_COEFF) * labels)
        acc2 = t.gt((1 + ACCURACY_COEFF) * labels, outputs)
        acc = t.logical_and(acc1, acc2)
        trainAcc = t.sum(acc)
        finalAcc = trainAcc/BATCH_SIZE

        # Printing and plotting results:
        if not round(0.1*EPOCHS) == 0:
            if epoch % round(0.1*EPOCHS) == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()}, Accuracy: {finalAcc}')
                inputs4show = inputs.detach().numpy()
                outputs4show = outputs.detach().numpy()
                #plt.plot(inputs4show, outputs4show)
                #plt.plot(inputs4show, labels)
                #plt.show()

        if epoch == EPOCHS - 1:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item()}, Accuracy: {finalAcc}')
            inputs4show = inputs.detach().numpy()
            outputs4show = outputs.detach().numpy()
            plt.plot(inputs4show, outputs4show)
            plt.plot(inputs4show, labels)
            plt.show()



