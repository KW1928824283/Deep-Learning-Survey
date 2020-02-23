import torch
from torch import nn,optim
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
import time
# def net(X):
#     X = X.view((-1, num_inputs))
#     H = relu(torch.matmul(X, W1) + b1)
#     return torch.matmul(H, W2) + b2
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
	mnist_train,mnist_test = load_datas()

	train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=0)
	test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=0)
	return train_iter,test_iter
def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
	net = net.to(device)
	print("training on ",device)
	loss = torch.nn.CrossEntropyLoss()
	batch_count = 0
	for epoch in range(num_epochs):

		train_l_sum,train_acc_sum,n,start=0.0,0.0,0,time.time()
		for X,y in train_iter:
			X = X.to(device)
			y = y.to(device)
			X = X.unsqueeze(1)
			y_hat = net(X)
			l = loss(y_hat,y.long())
			optimizer.zero_grad()
			l.backward()
			optimizer.step()
			train_l_sum += l.cpu().item()
			train_acc_sum+=(y_hat.argmax(dim=1)==y.long()).sum().cpu().item()
			n+=y.shape[0]
			batch_count+=1
			test_acc = evaluate_accuracy(test_iter,net)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f,time %.1f sec'% (epoch + 1, train_l_sum / batch_count,train_acc_sum / n, test_acc, time.time() - start))
def evaluate_accuracy(data_iter,net,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
	acc_sum,n = 0.0,0
	with torch.no_grad():
		for X,y in data_iter:
			if isinstance(net,torch.nn.Module):
				net.eval()
				X = X.unsqueeze(1)
				acc_sum+=(net(X.to(device)).argmax(dim=1)==y.long().to(device)).float().sum().cpu().item()
				net.train()
			else:
				if ('is_training' in net.__code__.co_varnames):
					acc_sum+=(net(X,is_training=False).argmax(dim=1)==y.long()).float().sum().item()
				else:
					acc_sum+=(net(X).argmax(dim=1)==y.long()).float().sum().item()
			n+=y.shape[0]
	return acc_sum/n
class LeNet(nn.Module):
	"""docstring for LeNet"""
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1,6,5),
			nn.Sigmoid(),
			nn.MaxPool2d(2,2),
			nn.Conv2d(6,16,5),
			nn.Sigmoid(),
			nn.MaxPool2d(2,2)

		)
		self.fc = nn.Sequential(
			nn.Linear(16*4*4,120),
			nn.Sigmoid(),
			nn.Linear(120,84),
			nn.Sigmoid(),
			nn.Linear(84,10)
		)
	def forward(self,img):
		feature = self.conv(img)
		output = self.fc(feature.view(img.shape[0],-1))
		return output
if __name__ == '__main__':
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net = LeNet()
	print(net)
	batch_size = 256
	train_iter,test_iter = load_data_fashion_mnist(batch_size)
	lr,num_epochs = 0.001,20
	optimizer = torch.optim.Adam(net.parameters(),lr = lr)
	train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)

		