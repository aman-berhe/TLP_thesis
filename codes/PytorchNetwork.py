import torch
import torch.nn as nn
import torchvision

class NeuralNet(nn.Module):
	"""A Neural Network with a hidden layer"""
	def __init__(self, input_size,hidden_size,output_size):
		super(NeuralNet, self).__init__()
		self.layer1 = nn.Linear(input_size, hidden_size)
		self.layer2 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		output = self.layer1(x)
		output = self.relu(output)
		output = self.layer2(output)
        return output
