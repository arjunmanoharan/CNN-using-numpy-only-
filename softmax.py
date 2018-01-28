import numpy as np
import sys
import matplotlib.pyplot as plt


def createData():
	N = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in xrange(K):
	  ix = range(N*j,N*(j+1))
	  r = np.linspace(0.0,1,N) # radius
	  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
	  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	  y[ix] = j
	# lets visualize the data:
	#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	#plt.show()
	return X,y




def _init_():
	[inputData,label]=createData()
	dimension=inputData.shape[0]	
	number_classes=inputData.shape[1]
	weight=0.01*np.random.randn(2,3)
	bias=np.random.randn(number_classes,1)
	#calculateScore(weight,inputData,bias,label)
	return weight,bias,inputData,label

def calculateScore(weight,inputData,bias,label):
	#print(inputData.shape)
	#print(weight.shape)
	lambdha=0.0001
	score=np.dot(inputData,weight)
	exp=np.exp(score)
	data_loss=0
	probability=exp/np.sum(exp,axis=1, keepdims=True)
	for i in range(300):
		data_loss+=-np.log(probability[i,label[i]])
	#loss=-np.log(probability)
	#print(probability)
	
	total_loss=(data_loss/inputData.shape[0])+0.5*lambdha*np.sum(weight*weight)
	print("loss=",total_loss)
	return probability


def gradientDescent(weight,inputData,label,bias):
	stepsize=1e-0
	probability=calculateScore(weight,inputData,bias,label)
	dscore=probability    
	for i in range(300):
		dscore[i,label[i]]=probability[i, 	label[i]]-1    
	
	dweight=np.dot(inputData.transpose(),dscore)
	dweight/=300
	dweight+=0.0001*weight	

	weight+=-stepsize*dweight
	return weight

def train_model(weight,bias,inputData,label):
	for i in range(100000):
		weight=gradientDescent(weight,inputData,label,bias)
	#print(calculateScore(weight,inputData,bias,label))

	return weight

def accuracy(inputData,weight,label):
	score=np.dot(inputData,weight)
	predicted_class = np.argmax(score, axis=1)
	print 'training accuracy: %.2f' % (np.mean(predicted_class == label))

[weight,bias,inputData,label]=_init_()
weight=train_model(weight,bias,inputData,label)
accuracy(inputData,weight,label)

