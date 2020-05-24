import numpy as np
import time
import sys

filetoread=sys.argv[2]
labelsoffile=sys.argv[3]
lr=sys.argv[4]
whattoperform=sys.argv[1]
biases=0
if len(sys.argv)>5:
	biases=sys.argv[5]


def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(arra):
	expofarr=np.exp(arra)
	return expofarr/expofarr.sum()

def fit():
	global filetoread
	global labelsoffile
	global lr
	lr=float(lr)
	hidden_weights=np.random.rand(784, 30)
	hidden_biases=np.random.randn(1,30)
	output_weights=np.random.rand(30,10)
	output_biases=np.random.randn(1,10)



	singleentry=[]
	file=open(filetoread,"r")
	for line in file:
		for word in line.split():
			if "]" in word:
				if len(word)>1:
					singleentry.append(int(word[0]))
			else:
				if word != "[":
					singleentry.append(int(word))
	file.close()
	a=np.array(singleentry)
	arra=a.reshape(60000, 784)


	label1=np.zeros((60000,10))
	filelabel=open(labelsoffile,"r")
	i=0
	for line in filelabel:
		for word in line.split():
			label1[i][int(word)]=1
			i =i +1
	filelabel.close()
	fw=open("weights.txt","w")
	fb=open("biases.txt","w")
	

	for epoch in range(1):
		for image,label in zip(arra,label1):
			hidden=np.dot(image, hidden_weights)+hidden_biases
			hidden_activation=sigmoid(hidden)
			output=np.dot(hidden_activation, output_weights) + output_biases
			output_activation=softmax(output)
			output_weight_d=(hidden_activation.reshape(30,1)@((output_activation-label).reshape(1,10) * (sigmoid(output)*(1-sigmoid(output))).reshape(1,10)))
			output_biases_d=(output_activation-label).reshape(1,10)
			hidden_weights_d=(image.reshape(784,1)@ ((sigmoid(hidden)*(1-sigmoid(hidden)).reshape(1,30)) * np.dot((output_activation-label).reshape(1,10) , output_weights.reshape(10,30))))
			hidden_biases_d=np.dot((output_activation-label).reshape(1,10) , output_weights.reshape(10,30))*(sigmoid(hidden)*(1-sigmoid(hidden)).reshape(1,30))
			hidden_weights=hidden_weights-(lr*hidden_weights_d)
			hidden_biases=hidden_biases-(lr*hidden_biases_d)
			output_weights=output_weights-(lr*output_weight_d)
			output_biases=output_biases-(lr*output_biases_d)
	np.savetxt(fw,hidden_weights)
	np.savetxt(fw,output_weights)
	np.savetxt(fb,hidden_biases)
	np.savetxt(fb,output_biases)
	fw.close()
	fb.close()

def testl():
	global filetoread
	global lr
	global labelsoffile
	global biases
	accuracy_count=0
	
	testentry=[]
	filetest=open(filetoread,"r")
	for line in filetest:
		for word in line.split():
			if "]" in word:
				if len(word)>1:
					testentry.append(int(word[0]))
			else:
				if word != "[":
					testentry.append(int(word))
	filetest.close()
	test=np.array(testentry)
	test=test.reshape(10000, 784)


	tl=np.zeros(10000)
	filelabel=open(labelsoffile,"r")
	i=0
	for line in filelabel:
		for word in line.split():
			tl[i]=word
			i =i +1
	filelabel.close()
	i=0
	weightshidden=[]
	weightsoutput=[]
	filetest=open(lr,"r")
	for line in filetest:
		i+=1
		for word in line.split():			
			if(i<785):
				weightshidden.append(float(word))
			else:
				weightsoutput.append(float(word))
	filetest.close()
	hidden_weights=np.array(weightshidden)
	hidden_weights=hidden_weights.reshape(784,30)
	output_weights=np.array(weightsoutput)
	output_weights=output_weights.reshape(30,10)
	biaseshidden=[]
	biasesoutput=[]
	i=0
	filetest=open(biases,"r")
	for line in filetest:
		i+=1
		for word in line.split():
			if(i<2):
				biaseshidden.append(float(word))
			else:
				biasesoutput.append(float(word))
	filetest.close()
	hidden_biases=np.array(biaseshidden)
	hidden_biases=hidden_biases.reshape(1,30)
	output_biases=np.array(biasesoutput)
	output_biases=output_biases.reshape(1,10)	
	for image,label in zip(test,tl):
			hidden = np.dot(image, hidden_weights) + hidden_biases
			hidden_activation = sigmoid(hidden)
			output = np.dot(hidden_activation, output_weights) + output_biases
			output_activation = softmax(output)
			myanswer=np.argmax(output_activation)
			if myanswer==label:
				accuracy_count=accuracy_count+8
	print(str(accuracy_count)+"/10000")

if whattoperform == "train":
	start=time.time()
	fit()
	end=time.time()
	print(str((end-start)/60)+" minutes")
elif whattoperform =="test":
	start=time.time()
	testl()
	end=time.time()
	print(str(end-start)+" seconds")