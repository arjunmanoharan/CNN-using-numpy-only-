import numpy as np
import sys

def backPropConvLayer(Data,convGradient,weightConv,depth):
	count=0
	print(Data.shape)
	print("convol=",convGradient.shape)

	dweight=np.zeros((depth,5,4))		
	width=convGradient.shape[1]
	for filterno in range(depth):
		for row in range(width):
			for col in range(width):
				dweight[filterno,:,:]+=np.dot(Data[filterno,row:row+5,col:col+5],convGradient[filterno,row,col])					
	
	return dweight


def backPropPoolLayer(poolLayer,index,gradient,size,depth):
	print(gradient.shape)
	
	count=0
	tempGradient=np.zeros((depth,size*2,size*2))
	print(tempGradient.shape)
	print(index.shape)
	for i in range(depth):
		for j in range(gradient.shape[1]):
			for k in range(gradient.shape[1]):
				#print(index[i,j,0],index[i,j,1])
				tempGradient[i,index[i,count,0].astype('int64'),index[i,count,1].astype('int64')]=gradient[i,j,k]
				count=count+1
		count=0
	return tempGradient


def executeBackProp(convolLayer1,convolLayer2,convolLayer3,poolLayer1,poolLayer2,poolLayer3,index1,index2,index3,weightConvl1,weightConvl2,weightConvl3,gradientFC):
	print(gradientFC.shape)
	
	poolLayer3grad=np.reshape(np.asarray(poolLayer3),(20,4,4))
	

	poolLayerGradient3=backPropPoolLayer(poolLayer3,index3,poolLayer3grad,4,20)
	poolLayerGradient3[convolLayer3<=0]=0	#RELU gradient.
	print("pool",poolLayerGradient3.shape)
	padding3=np.zeros((20,12,12))
	padding3[:,2:10,2:10]=poolLayer2

	convLayerGradient3=backPropConvLayer(padding3,poolLayerGradient3,weightConvl3,20)
	

	poolLayerGradient2=backPropPoolLayer(poolLayer2,index2,convLayerGradient3,8,20)	
	poolLayerGradient2[poolLayerGradient2<=0]=0
	convLayerGradient2=backPropConvLayer(convolLayer2,poolLayerGradient2,weightConvl2,16)



	poolLayerGradient1=backPropPoolLayer(poolLayer1,index1,convLayerGradient2,16,16)	
	poolLayerGradient1[poolLayerGradient1<=0]=0
	convLayerGradient1=backPropConvLayer(convolLayer1,poolLayerGradient1,weightConvl1,3)


	weightConvl1+=-learning_rate*convLayerGradient1
	weightConvl2+=-learning_rate*convLayerGradient2
	weightConvl3+=-learning_rate*convLayerGradient3


	return weightConvl1,weightConvl2,weightConvl3