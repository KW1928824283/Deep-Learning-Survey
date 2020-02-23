import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import gzip
from IPython import display
from matplotlib import pyplot as plt 
import torch.utils.data as Data

def sgd(params,lr,batch_size):
	for param in params:
		param.data-=lr*param.grad / batch_size
def evaluate_accuracy(data_iter,net):
	acc_sum,n = 0.0,0
	for X,y in data_iter:
		acc_sum += (net(X).argmax(dim=1)==y.long()).float().sum().item()

		n+=y.shape[0]
	return acc_sum/n
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params = None,lr = None,optimizer=None):
	for epoch in range(num_epochs):
		train_l_sum,train_acc_sum,n=0.0,0.0,0
		for X,y in train_iter:
			y_hat = net(X)
			l = loss(y_hat,y.long()).sum()
			if optimizer is not None:
				optimizer.zero_grad()
			elif params is not None and params[0].grad is not None:
				for param in params:
					param.grad.data.zero_()
			l.backward()
			if optimizer is None:
				sgd(params, lr, batch_size)
			else:
				optimizer.step()
			train_l_sum += l.item()
			train_acc_sum+=(y_hat.argmax(dim=1)==y.long()).sum().item()
			n+=y.shape[0]
			test_acc = evaluate_accuracy(test_iter,net)
		print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f'%(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))

def net(X):

    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

def load_datas():
	paths = [
		'./datasets/train-labels-idx1-ubyte.gz',
		'./datasets/train-images-idx3-ubyte.gz',
		'./datasets/t10k-labels-idx1-ubyte.gz',
		'./datasets/t10k-images-idx3-ubyte.gz'
	]
	with gzip.open(paths[0],'rb') as lbpath:
		y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)

	with gzip.open(paths[1],'rb') as imgpath:
		x_train = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(y_train),28,28)

	x_train = x_train/255.0

	with gzip.open(paths[2],'rb') as lbpath:
		y_test = np.frombuffer(lbpath.read(),np.uint8,offset=8)
	with gzip.open(paths[3],'rb') as imgpath:
		x_test = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(y_test),28,28)
	x_test = x_test/255.0

	train_images = torch.tensor(x_train,dtype =  torch.float)
	train_labels = torch.tensor(y_train,dtype =  torch.float)
	test_images  = torch.tensor(x_test,dtype =  torch.float)
	test_labels  = torch.tensor(y_test,dtype =  torch.float)

	train_mnist  = Data.TensorDataset(train_images,train_labels)
	test_mnist   = Data.TensorDataset(test_images,test_labels)
	return train_mnist,test_mnist

def load_data_fashion_mnist(batch_size,resize=None):
	trans = []
	if resize:
		train.append(torchvision.transforms.Resize(size = resize))
	trans.append(torchvision.transforms.ToTensor())
	transform = torchvision.transforms.Compose(trans)
	print('hello')
	mnist_train,mnist_test = load_datas()

	train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=0)
	test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=0)
	return train_iter,test_iter
def use_svg_display():
	display.set_matplotlib_formats('svg')
def show_fashion_mnist(images,labels):
	use_svg_display()
	_,figs = plt.subplots(1,len(images),figsize=(12,12))
	for f,img,lbl in zip(figs,images,labels):
		f.imshow(img.view((28,28)).numpy())
		f.set_title(lbl)
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
	plt.show()
def get_fashion_mnist_lables(labels):
	text_lables = ['t-shirt', 'trouser', 'pullover', 'dress','coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle','boot']
	return [text_lables[int(i)] for i in labels]
if __name__ == '__main__':
	batch_size = 256
	train_iter, test_iter = load_data_fashion_mnist(batch_size)
	num_inputs, num_outputs, num_hiddens = 784, 10, 256
	W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
	b1 = torch.zeros(num_hiddens, dtype=torch.float)
	W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
	b2 = torch.zeros(num_outputs, dtype=torch.float)
	params = [W1, b1, W2, b2]
	for param in params:
	    param.requires_grad_(requires_grad=True)	
	loss = torch.nn.CrossEntropyLoss()
	num_epochs, lr = 20, 100.0
	train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
	X,y = iter(test_iter).next()
	true_labels = get_fashion_mnist_lables(y.numpy())
	pred_labels  = get_fashion_mnist_lables(net(X).argmax(dim=1).numpy())
	titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
	show_fashion_mnist(X[0:9], titles[0:9])