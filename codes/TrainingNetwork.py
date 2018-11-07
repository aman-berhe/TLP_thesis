import numpy as np
from sklean import
from sklean.model_selsection import train_test_split
import torch
import torch.nn as nn
import torchvision

input_size = 784
hidden_size = 500
output_size = 19
num_epochs = 5

learning_rate = 0.001

model = NeuralNet(input_size,hidden_size, output_size)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def loadData(dataFile,labelsFile):
    """
    load data and split it into training, validation and test data set
    """
    x=np.load(file=DataFile)
    y=np.load(file=labelsFile)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


def trainingNetwork()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
    	for i, (images,labels) in enumerate(train_loader):
    		images = images.reshape(-1,28*28)
    		show_batch(images)
    		out = model(images)
    		loss = lossFunction(out,labels)

    		optimizer.zero_grad()
    		loss.backward()
    		optimizer.step()

    		if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
