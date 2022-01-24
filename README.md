# PyNUTS
This project implements bayesian modeling and probabilistic machine learning with python using No-U-Turn sampler (NUTS) (Adaptively Setting PathLengths in Hamiltonian Monte Carlo,Journal of Machine Learning Reserach 15(2014),1593-1623))
      
##########################################################################    
Author: Junjun Guo([HomePage](https://github.com/Junjun1guo))    
E-mail: guojj@tongji.edu.cn/guojj_ce@163.com    
Environemet: Successfully excucted in python 3.8    
##########################################################################
______
## required modules
[numpy](https://numpy.org/)   [matplotlib](https://matplotlib.org/)   [arviz](https://arviz-devs.github.io/arviz/)   [tqdm](https://github.com/tqdm/tqdm)   [pytorch](https://pytorch.org/)   [records](https://github.com/kennethreitz/records)
______
## Tutorials
1. Download the zip file
2. Make sure the required modules are installed, and run the examples in pyNUTSExample.py file.
3. Currently, the module provides Markov chain Monte Corlo (MCMC) sampling for multivariate normal distribution,generalized linear regression ang logistic regression.    

## Example 1. Multivariate normal distribution 
```python
import numpy as np
import matplotlib.pyplot as plt
from PyNUTS import MultiNormal

meanArray = np.array([[1.2,-3.4]])
covArray = np.array([[2.1,-1.2],[-1.2,5.2]])
numSample = 5000
numBurning = 5000
initalThetaArray = np.random.normal(0, 1, 2)
testInstance=MultiNormal(meanArray, covArray, numSample, numBurning, initalThetaArray)
nutsSamples=testInstance.sample()
print('Mean')
print(np.mean(nutsSamples, axis=0))
print('Stddev')
print(np.std(nutsSamples, axis=0))
sampleValues = np.random.multivariate_normal(meanArray[0, :],covArray, size=numSample)
plt.plot(sampleValues[:, 0], sampleValues[:, 1], 'b.')
plt.plot(nutsSamples[:, 0], nutsSamples[:, 1], 'r+')
plt.savefig("multivariateSample.jpg")
plt.savefig("multivariateSample.eps")
plt.show()
plt.subplot(2, 2, 1)
plt.hist(nutsSamples[:, 0], bins=50,color='red')
plt.xlabel("x1-nutsSamples")
plt.subplot(2, 2, 2)
plt.hist(sampleValues[:, 0], bins=50, color='blue')
plt.xlabel("x1-npSamples")
plt.subplot(2, 2, 3)
plt.hist(nutsSamples[:, 1], bins=50,color='red')
plt.xlabel("x2-nutsSamples")
plt.subplot(2, 2, 4)
plt.hist(sampleValues[:, 1], bins=50, color='blue')
plt.xlabel("x2-npSamples")
plt.savefig("multiNormalCompare.jpg")
plt.savefig("multiNormalCompare.eps")
plt.show()
```
<img src="https://github.com/Junjun1guo/PyNUTS/blob/main/multivariateSample.png" width =45% height =45% div align="left"><img src="https://github.com/Junjun1guo/PyNUTS/blob/main/multiNormalCompare.png" width =45% height =45% div align="left">　　

## Example 2. Generalized linear regression  

```python
import numpy as np
import matplotlib.pyplot as plt
from PyNUTS import LinearRegression

n = 100
alpha = 2
beta = 5
gamma=4
sigma = 0.2
x1 = np.linspace(0, 1, n)
x2=np.array([np.random.normal(0, sigma,1)[0]*each for each in x1])
y = alpha + beta * x1 + gamma * x2+np.random.normal(0, sigma, n)
x=[[1.0,x1[i],x2[i]] for i in range(n)]
numSampling=15000
numBurning=20000
linearInstance=LinearRegression(x,y,numSampling,numBurning)
linearInstance.sample()
linearInstance.plotTrace()
linearInstance.plotPosterior()
linearInstance.plotAutoCorr()
linearInstance.summary()
```  
<img src="https://github.com/Junjun1guo/PyNUTS/blob/main/plotTrace_linear.png" width =45% height =45% div align="left">
<img src="https://github.com/Junjun1guo/PyNUTS/blob/main/plotPosterior_linear.png" width =45% height =45% div align="center">

## Example 3. Logistic regression

```python
xy=np.loadtxt("logisticRegression.txt")
x=list(xy[:,0])
y=list(xy[:,1])
x=[[1.0,np.log(each)] for each in x]
numSampling=20000
numBurning=5000
logisticInstance=LogisticRegression(x,y,numSampling,numBurning)
logisticInstance.sample()
logisticInstance.plotTrace()
logisticInstance.plotPosterior()
logisticInstance.plotAutoCorr()
logisticInstance.summary()
```   
<img src="https://github.com/Junjun1guo/PyNUTS/blob/main/plotTrace_logistic.png" width =45% height =45% div align="left">
<img src="https://github.com/Junjun1guo/PyNUTS/blob/main/plotPosterior_logistic.png" width =50% height =50% div align="left">




