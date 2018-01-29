import numpy as np
import sys
import NeuralNet as NNet
import cPickle 
import forwardPass as fPass
import backPropagate as backProp
import os 
RECEPTIVE_FIELD=5
STRIDE=1
PADDING=0
INPUT_FRAME=32
dataGradient=np.zeros((5,5,3,784))
output_frame_size=(INPUT_FRAME-RECEPTIVE_FIELD+(2*PADDING))/STRIDE+1
pathname= os.path.dirname(sys.argv[0])
def getData():
	with open(pathname+'\cifar-10-batches-py\data_batch_1', 'rb') as fo:
		
		dict = cPickle.load(fo)
	
	inputData=dict['data']
	labels=dict['labels']
	
	labels=np.matrix(labels)
	inputData=np.matrix(inputData)
	print(labels.shape)
	print(inputData.shape)

	return inputData,labels.transpose()

def setParameter():
	[inputData,labels]=getData()
	weightFC1=0.01*np.random.randn(320,100)
	weightFC2=0.01*np.random.randn(100,10)
	biasFC1=np.random.randn(100,1)
	biasFC2=np.random.randn(10,1)

	return weightFC1,weightFC2,biasFC1,biasFC2,inputData,labels

def forwardPropogation(inputData,labels,weightFC1,weightFC2,biasFC1,biasFC2):
	

	inputFC=fPass.combineLayers(inputData)
	rows=inputFC.shape[0]

	
	
	[total_Loss,probability,layer1_activation]=NNet.calculateLoss(inputFC,weightFC1,weightFC2,biasFC1,biasFC2,labels)

	print("los=====",total_Loss)
	return weightFC1,inputFC,dataGradient,probability,layer1_activation
	


def accuracy(weight1,weight2,bias2,bias1,dweight1,Data,labels):
	count=0

	hidden_layer = np.maximum(0, np.dot(weight1.transpose(),Data) + bias1)
	scores = np.dot(weight2.transpose(),hidden_layer) + bias2
	exp=np.exp(scores)
	probability=exp/np.sum(exp,axis=0)
	
	for i in range(100):
		if(labels[i]==np.argmax(probability[:,i])):
			count=count+1
		
	print(count)



def trainingNeuralNet():
	[inputData,labels]=getData()
	weight1=0.001*np.random.randn(3072,3072)
	weight2=0.001*np.random.randn(3072,10)
	bias1=0.001*np.random.randn(3072,1)
	bias2=0.001*np.random.randn(10,1)
	Data=np.matrix(inputData[0:100]).transpose()
	[weight1,weight2,bias2,bias1,dweight1]=NNet.backProp(Data,weight1,weight2,bias1,bias2,labels)
	accuracy(weight1,weight2,bias2,bias1,dweight1,Data,labels)
	
	


def backPropagation(Data,inputData,weightConv,weightFC1,weightFC2,bias1,bias2,labels,probability,layer1_activation):
	
	tempData=np.reshape(Data[0],(32,32,-1))
	
	for i in range(1):
		count=0		
		[weight1,weight2,bias2,bias1,dweightFC1,dhidden]=NNet.backProp(inputData.transpose(),weightFC1,weightFC2,bias1,bias2,labels,probability,layer1_activation)
		weightFC1-=dweightFC1
		convGradient=weight1*(weight2*dhidden)
		dconvGradient=np.reshape(convGradient,(28,28))
		dweight=np.zeros((5,5,3))	
		for k in range(3):
			data=tempData[:,:,k]
			for j in range(5):
				for i in range(5):
					lower_limit=i				
					upper_limit=lower_limit+28
					
					dweight[j,i,k]=np.sum(np.multiply(weight[:,:],data[j:j+RECEPTIVE_FIELD,lower_limit:upper_limit]))	
						

		
		weightConv=backPropConvLayer(Data,convGradient,weightConv)

	return weightConv,weightFC1
	


def trainCNN():
	[weightFC1,weightFC2,bias1,bias2,inputData,labels]=setParameter()
	
	#have to loop
	[inputFC,convolLayer1,convolLayer2,convolLayer3,poolLayer1,poolLayer2,poolLayer3,index1,index2,index3,weightConvl1,weightConvl2,weightConvl3]=fPass.combineLayers(inputData,None,None,None)	
	[weightFC1,weightFC2,bias2,bias1,dweight1,dprobability]=NNet.executeNNet(inputFC.transpose(),weightFC1,weightFC2,bias1,bias2,labels)
	gradientFC=weightFC1*(weightFC2*dprobability)
	[weightConvl1,weightConvl2,weightConvl3]=backProp.executeBackProp(convolLayer1,convolLayer2,convolLayer3,poolLayer1,poolLayer2,poolLayer3,index1,index2,index3,weightConvl1,weightConvl2,weightConvl3,gradientFC)

trainCNN()
