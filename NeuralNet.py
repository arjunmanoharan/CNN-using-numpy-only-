import numpy as np
import sys


def calculateLoss(inputData,weight1,weight2,bias1,bias2,label):
	data_Loss=0
	lambdha=1e-1
	layer1=np.dot(weight1.transpose(),inputData)+bias1
	layer1_activation=np.maximum(0,layer1)
	output_layer=np.dot(weight2.transpose(),layer1_activation)+bias2
	number_iterations=inputData.shape[1]
	print(number_iterations)
	exp=np.exp(output_layer)
	probability=exp/np.sum(exp,axis=0)
	
	
	for i in range(0,number_iterations-1):
		data_Loss+=-np.log(probability[label[i],i])
		
	regularization_Loss=0.5*lambdha*np.sum(weight2*weight2)
	data_Loss/=number_iterations
	total_Loss=data_Loss+regularization_Loss
	print("loss=",total_Loss)

	return total_Loss,probability,layer1_activation


def backProp(inputData,weight1,weight2,bias1,bias2,label,probability,layer1_activation):
	learning_rate=0.0000001
	dprobability=probability  
	number_iterations=inputData.shape[1]

	for i in range(number_iterations):
		dprobability[label[i],0]=probability[label[i],0]-1

	dprobability/=number_iterations
	
	dweight2=np.dot(layer1_activation,dprobability.transpose())
	weight2-=learning_rate*dweight2
	
	temp=np.dot(weight2,dprobability)
	temp[temp <= 0]=0
 	
	dweight1=np.dot(inputData,temp.transpose())
	weight1-=learning_rate*dweight1
	
	dbias2 = np.sum(dprobability, axis=1)
	dbias1 = np.sum(temp, axis=1)
	bias1-=learning_rate*dbias1
	bias2-=learning_rate*dbias2
	
	return weight1,weight2,bias2,bias1,dweight1,dprobability

def executeNNet(inputData,weight1,weight2,bias1,bias2,label):
	[total_Loss,probability,layer1_activation]=calculateLoss(inputData,weight1,weight2,bias1,bias2,label)
	[weight1,weight2,bias2,bias1,dweight1,dprobability]=backProp(inputData,weight1,weight2,bias1,bias2,label,probability,layer1_activation)

	return weight1,weight2,bias2,bias1,dweight1,dprobability