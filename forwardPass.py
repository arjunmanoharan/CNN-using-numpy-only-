import numpy as np
import sys
import cPickle
from numpy import unravel_index
RECEPTIVE_FIELD=5
STRIDE=1
PADDING=0
INPUT_FRAME=32


def getData():
	with open("E:\Arjun\work\iit\course\dl\project\cnn\cs231\cifar-10-batches-py\data_batch_1", 'rb') as fo:
		dict = cPickle.load(fo)
	inputData=dict["data"]
	labels=dict["labels"]
	return inputData,labels

def createWeights():
	
	weightConvl1=np.zeros((16,3,5,5))
	weightConvl1=defineWeights(5,3,16)
	weightConvl2=np.zeros((20,16,5,5))
	weightConvl2=defineWeights(5,16,20)
	weightConvl3=np.zeros((20,20,5,5))
	weightConvl3=defineWeights(5,20,20)

	return weightConvl1,weightConvl2,weightConvl3

def generateConvLayer(Data,weight):
	INPUT_FRAME=Data.shape[1]
	output_frame_size=INPUT_FRAME-RECEPTIVE_FIELD+1	
	count=0	
	convol=np.zeros((output_frame_size,output_frame_size))
	index=np.zeros((output_frame_size*output_frame_size,2))
	for j in range(output_frame_size):
		for i in range(output_frame_size):
			lower_limit=STRIDE*i				
			upper_limit=lower_limit+RECEPTIVE_FIELD
			
			convol[j,i]=np.sum(np.multiply(weight[:,:],Data[:,j:j+RECEPTIVE_FIELD,lower_limit:upper_limit]))	
			
			if(convol[j,i]>0):
				index[count,0]=j
				index[count,1]=i
			count=count+1		

	convolRELU=np.maximum(0,convol)
	#print(index)
	return convolRELU	


def generatePoolingLayer(Data,size):
	INPUT_FRAME=Data.shape[1]
	count=0
	output_frame_size=INPUT_FRAME/size
	poolLayer=np.zeros((output_frame_size,output_frame_size))
	index=np.zeros((output_frame_size*output_frame_size,2))

	for j in range(output_frame_size):
		for i in range(output_frame_size):
			lower_limit=2*i				
			upper_limit=lower_limit+size
			tempData=Data[j:j+size,lower_limit:upper_limit]	
			#print("temp=",tempData.shape)
			[index[count,0],index[count,1]]=unravel_index(tempData.argmax(),tempData.shape)
			
			poolLayer[j,i]=np.max(tempData)
		count=count+1	
	return poolLayer,index

def defineWeights(size,depth,number_filters):
	weight=np.zeros((number_filters,depth,size,size))
	for i in range(number_filters):
		weight[i,:,:,:]=0.001*np.random.randn(depth,size,size)

	return weight



def combineLayers(inputData,weightConvl1,weightConvl2,weightConvl3):
	print(inputData[0].shape)
	rawData=np.reshape(np.asarray(inputData[0]),(3,32,32))
	padding1=np.zeros((3,36,36))
	padding1[:,2:34,2:34]=rawData

	if not weightConvl1:
		[weightConvl1,weightConvl2,weightConvl3]=createWeights()


	poolLayer1=np.zeros((16,16,16))
	convolLayer1=np.zeros((16,32,32))
	convolLayer2=np.zeros((20,16,16))
	convolLayer3=np.zeros((20,8,8))
	poolLayer2=np.zeros((20,8,8))
	poolLayer3=np.zeros((20,4,4))
	

	index1=np.zeros((16,256,2))
	index2=np.zeros((20,64,2))
	index3=np.zeros((20,16,2))
	
	for i in range(16):
		convolLayer1[i,:,:]=generateConvLayer(padding1,weightConvl1[i,:,:,:])
		[poolLayer1[i,:,:],index1[i,:,:]]=generatePoolingLayer(convolLayer1[i,:,:],2)
			
	padding2=np.zeros((16,20,20))
	padding2[:,2:18,2:18]=poolLayer1

	for i in range(20):
		convolLayer2[i,:,:]=generateConvLayer(padding2,weightConvl2[i,:,:,:])
		[poolLayer2[i,:,:],index2[i,:,:]]=generatePoolingLayer(convolLayer2[i,:,:],2)
	
	padding3=np.zeros((20,12,12))
	padding3[:,2:10,2:10]=poolLayer2

	for i in range(20):
		convolLayer3[i,:,:]=generateConvLayer(padding3,weightConvl3[i,:,:,:])
		[poolLayer3[i,:,:],index3[i,:,:]]=generatePoolingLayer(convolLayer3[i,:,:],2)
	print(poolLayer3.shape)
	#print(poolLayer3)
	output=poolLayer3
	inputFC=np.zeros((160,1))
	inputFC=np.matrix(output.ravel())	

	return inputFC,convolLayer1,convolLayer2,convolLayer3,poolLayer1,poolLayer2,poolLayer3,index1,index2,index3,weightConvl1,weightConvl2,weightConvl3
