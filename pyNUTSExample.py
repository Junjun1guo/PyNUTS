#-*-coding: UTF-8-*-
#########################################################################
#   Author: Junjun Guo
#   E-mail: guojj@tongji.edu.cn/guojj_ce@163.com
#   Environment: Successfully executed in python 3.8
#   Date:1/10/2022
############################################################################
import numpy as np
import matplotlib.pyplot as plt
from PyNUTS import MultiNormal,LinearRegression,LogisticRegression
############################################################################
#########################---multiNormal example---##########################
# meanArray = np.array([[1.2,-3.4]])
# covArray = np.array([[2.1,-1.2],[-1.2,5.2]])
# numSample = 5000
# numBurning = 5000
# initalThetaArray = np.random.normal(0, 1, 2)
# testInstance=MultiNormal(meanArray, covArray, numSample, numBurning, initalThetaArray)
# nutsSamples=testInstance.sample()
# print('Mean')
# print(np.mean(nutsSamples, axis=0))
# print('Stddev')
# print(np.std(nutsSamples, axis=0))
# sampleValues = np.random.multivariate_normal(meanArray[0, :],covArray, size=numSample)
# plt.plot(sampleValues[:, 0], sampleValues[:, 1], 'b.')
# plt.plot(nutsSamples[:, 0], nutsSamples[:, 1], 'r+')
# plt.savefig("multivariateSample.jpg")
# plt.savefig("multivariateSample.eps")
# plt.show()
# plt.subplot(2, 2, 1)
# plt.hist(nutsSamples[:, 0], bins=50,color='red')
# plt.xlabel("x1-nutsSamples")
# plt.subplot(2, 2, 2)
# plt.hist(sampleValues[:, 0], bins=50, color='blue')
# plt.xlabel("x1-npSamples")
# plt.subplot(2, 2, 3)
# plt.hist(nutsSamples[:, 1], bins=50,color='red')
# plt.xlabel("x2-nutsSamples")
# plt.subplot(2, 2, 4)
# plt.hist(sampleValues[:, 1], bins=50, color='blue')
# plt.xlabel("x2-npSamples")
# plt.savefig("multiNormalCompare.jpg")
# plt.savefig("multiNormalCompare.eps")
# plt.show()
############################################################################
##############---generalized linear regression example---############
# n = 100
# alpha = 2
# beta = 5
# gamma=4
# sigma = 0.2
# x1 = np.linspace(0, 1, n)
# x2=np.array([np.random.normal(0, sigma,1)[0]*each for each in x1])
# y = alpha + beta * x1 + gamma * x2+np.random.normal(0, sigma, n)
# x=[[1.0,x1[i],x2[i]] for i in range(n)]
# numSampling=15000
# numBurning=20000
# linearInstance=LinearRegression(x,y,numSampling,numBurning)
# linearInstance.sample()
# linearInstance.plotTrace()
# linearInstance.plotPosterior()
# linearInstance.plotAutoCorr()
# linearInstance.summary()
############################################################################
##############---logistic regression example---#############################
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
############################################################################
