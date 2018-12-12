import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
BATCH_SIZE = 1
N, D_in, H, D_out = BATCH_SIZE, 784, 200, 10
n_epochs = 20



# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.ToTensor()

# Load and transform data
trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()

print('Labels: ', labels)
print('Batch shape: ', images.size())
show_batch(images)
plt.show()



dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # run on GPU

class Network(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(Network, self).__init__()
		# random weights and biases.
		# set requires_grad=True so we CAN compute gradients with respect
		# to these tensors during backprop.
		self.w1 = torch.randn(H, D_in, device=device, dtype=dtype, requires_grad=True)
		self.w2 = torch.randn(D_out, H, device=device, dtype=dtype, requires_grad=True)
		self.b1 = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
		self.b2 = torch.randn(D_out, device=device, dtype=dtype, requires_grad=True)

	def forward(self, x):
		# forward pass
		y_pred = F.linear(x, self.w1, self.b1)
		y_pred = F.relu(y_pred)
		y_pred = F.linear(y_pred, self.w2, self.b2)
		y_pred = F.log_softmax(y_pred, dim=1)

		return y_pred



model = Network(D_in, H, D_out)

def train(model, trainloader, n_epochs=2):
	learning_rate = .001
	for t in range(n_epochs):
		print("epoch: ",t)
		correct = 0
		for i, data in enumerate(trainloader):
			inputs, labels = data
			inputs, labels = Variable(inputs), Variable(labels)

			# reformat data
			inputs = inputs.squeeze().reshape(inputs.shape[0], inputs.shape[2]*inputs.shape[3])

			# forward pass
			y_pred = model(inputs)
			predicted = torch.argmax(y_pred.data, dim=1)
			correct += (predicted == labels).sum()

			# compute loss
			loss_func = nn.CrossEntropyLoss()
			loss = loss_func(y_pred, labels)

			# use autograd to compute backprop.
			# computes gradient of loss with respect to all tensors with requires_grad=True.
			# w1.grad, w2.grad, b1.grad, and b2.grad will be tensors of gradient of loss with respect to w1 and w2.
			loss.backward()
			
			# wrap in torch.no_grad() so we don't track weights in
			# autograd while updating them.
			with torch.no_grad():
				model.w1 -= learning_rate * model.w1.grad
				model.w2 -= learning_rate * model.w2.grad
				model.b1 -= learning_rate * model.b1.grad
				model.b2 -= learning_rate * model.b2.grad

				# zero out gradients for the next pass after updating them.
				model.w1.grad.zero_()
				model.w2.grad.zero_()
				model.b1.grad.zero_()
				model.b2.grad.zero_()

		print('Accuracy: ', 100 * correct / len(trainset))


def test(model, testloader, n):
	correct = 0
	for data in testloader:
		inputs, labels = data
		inputs, labels = Variable(inputs), Variable(labels)

		# reformat data
		inputs = inputs.squeeze().reshape(inputs.shape[0], inputs.shape[2]*inputs.shape[3])

		outputs = model(inputs)
		pred = torch.argmax(outputs.data, dim=1)
		correct += (pred == labels).sum()
	return 100 * correct / n


train(model, trainloader, n_epochs)
print('Test Data Accuracy: ', test(model, testloader, len(testset)))