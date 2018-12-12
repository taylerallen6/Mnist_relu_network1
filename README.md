# Mnist_relu_network1

This is an example of a fully connected neural network classifying the mnist dataset.

The network is written in python3 using <a href="https://github.com/pytorch">Pytorch</a> for tensor operations. It has 3 layers with a relu activation function in the hidden layer and a log softmax activation function in the output layer. Once trained the network has an accuracy of 96% on the training data and 93% accuracy on the test data. It's be no means perfect, but is a good experiment. Accuracy can be further improved by making ajustments to the batch size, learning rate, hidden layer size, and number of hidden layer. For example, the current hidden layer size is 200, but by increasing it to 300, the test data accuracy increases to 94%. Not much difference but it's a start.

<br/>

<h2>Install</h2>

1. You need python 3 to run the script. Make sure you have it be continuing. This will not work with python 2.
2. Create a virtual environment and activate it.
3. Run 'pip3 install torch torchvision'.
4. Clone this repository in your virtual environment.
5. In your terminal, navigate to the directory containing the pt_mnist_relu1.py file.
6. Run 'python3 pt_mnist_relu1.py'

<br/>

<h2>Code Explained</h2>

Note that this only explains parts of the code. It will not work if you simply copy each code snippets
I have listed. Please follw the install process.

Batch size = 1 <br/>
Input size = 784 (28 x 28 for each image) <br/>
Hidden size = 200 <br/>
output size = 10 (value for each digit) <br/>
number of epochs = 20 <br/>

```
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
BATCH_SIZE = 1
N, D_in, H, D_out = BATCH_SIZE, 784, 200, 10
n_epochs = 20
```

<br/>

<h3>Get mnist dataset</h3>

I will not go in depth with this code other than it loads in the mnist dataset for you to use. The data is diveded into a training set of 60,000 images and a test set of 10,000 images.
It then shows a few sample images based on the batch size. With the batch size currently set at 1, there with only be one image shown. If you increase the batch size, you will see more images displayed at once.
```
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
```

<br/>

<h3>The network model</h3>

The network model has two weight tensors and two bias tensors. Then in the forward function, there is a linear function, a relu function, another linear function, and finally a log softmax function. The first linear takes the inputs, first weights, and first biases. The relu takes the output of the first linear. The second linear takes the relu output, the second weights, and the second biases. Lastly, the log softmax takes the output of the second linear.

```
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
```

<br/>

I initialize the model using the variables defined above:
```
model = Network(D_in, H, D_out)
```

<br/>

<h3>Function to train the network</h3>

I then define a train function taking the model, the training data loader, and the number of epochs. It defines the learning rate = .001, then iterates through each epoch. For each epoch, it then iterates through the entire training set.
```
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
```

<br/>

<h4>Deeper explaining the train function</h4>

```
inputs, labels = data
inputs, labels = Variable(inputs), Variable(labels)

# reformat data
inputs = inputs.squeeze().reshape(inputs.shape[0], inputs.shape[2]*inputs.shape[3])
```
This defines the input data and the label data. Then, the input data is reshaped the be a tensor of [1, 784] instead of a tensor of [1, 1, 28, 28].

<br/>

```
# forward pass
y_pred = model(inputs)
predicted = torch.argmax(y_pred.data, dim=1)
correct += (predicted == labels).sum()
```
This calls the forward() function and calculates the prediction. Note, the forward() function is automatically called when calling model(inputs). The correct += (predicted == labels).sum() helps calculate the accuracy for the entire epoch.

<br/>

```
# compute loss
loss_func = nn.CrossEntropyLoss()
loss = loss_func(y_pred, labels)

# use autograd to compute backprop.
# computes gradient of loss with respect to all tensors with requires_grad=True.
# w1.grad, w2.grad, b1.grad, and b2.grad will be tensors of gradient of loss with respect to w1 and w2.
loss.backward()
```
This calculates the loss using cross entropy and computes the gradients using the .backward() method.

<br/>

```
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
```
This updates the weights and biases, then zeros the gradients so they can be computed again in the next iteration.

<br/>

<h3>Function to test the network</h3>

```
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
```
This is pretty much the same as the train() function but with two main differences. First, there is not epoch iteration. This is because you do not need to run the network over the test data more than once. Second, there is no loss calculation, weights and bias updating, and gradient zeroing. This is because there is no need to update the model while testing. You only want to test the model on what it learned while training. If it learns more while testing, the results will be inaccurate.

<br/>

<h3>Actually train and test the network</h3>

```
train(model, trainloader, n_epochs)
print('Test Data Accuracy: ', test(model, testloader, len(testset)))
```
The result prints the accuracy for each epoch of training and lastly, the test accuracy.
