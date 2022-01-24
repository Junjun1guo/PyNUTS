#-*-coding: UTF-8-*-
#########################################################################
#   Author: Junjun Guo
#   E-mail: guojj@tongji.edu.cn/guojj_ce@163.com
#   Environment: Successfully executed in python 3.8
#   Date:1/10/2022
#########################################################################
# purpose: This program implements the No-U-Turn Sampler(NUTS)
# proposed by Hoffman and Gelman(The No-U-Turn Sampler: Adaptively Setting Path
# Lengths in Hamiltonian Monte Carlo,Journal of Machine Learning Reserach 15(2014),1593-1623)
#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from tqdm.std import trange
import torch ##pip3 install torch torchvision torchaudio (cpu) https://pytorch.org/
import records
import warnings
warnings.filterwarnings('ignore')
#########################################################################
#########################################################################
__all__ = ['MultiNormal','LogisticRegression','LinearRegression']##can be called
class ArvizPlot():
    """
    This class implements the sampler plot using arviz moudle
    https://arviz-devs.github.io/arviz/
    """
    @staticmethod
    def loadData():
        """Load data from sqlite database"""
        dbPath = "samplesDB.db"
        saveInstance = SqliteDB(dbPath)
        samples = saveInstance.getSamples()
        returnSample = SqliteDB.strToListConvert(samples, "samples", "float")
        returnArray = np.array(returnSample)
        return returnArray

    @classmethod
    def plotTrace(cls,variableList):
        """
        Plot the trace of the samples
        Inputs:
            variableList(str list)-the parameters' name list ["beta_0","beta_1",...]
        """
        returnArray=ArvizPlot.loadData()
        dataDict={}
        for i1 in range(len(variableList)):
            dataDict[variableList[i1]]=returnArray[:,i1]
        az.style.use("arviz-whitegrid")
        az.plot_trace(dataDict,var_names=variableList)
        plt.savefig("plotTrace.jpg")
        plt.savefig("plotTrace.eps")
        plt.show()

    @classmethod
    def plotPosterior(cls,variableList,hdi_prob=.95):
        """
        Plot the posterior distribution of the sampled variable
        Inputs:
            variableList(str list)-the parameters' name list ["beta_0","beta_1",...]
            hdi_prob(float)-Plots highest density interval for chosen percentage of density.
        """
        returnArray=ArvizPlot.loadData()
        dataDict = {}
        for i1 in range(len(variableList)):
            dataDict[variableList[i1]] = returnArray[:, i1]
        az.style.use("arviz-whitegrid")
        if int(len(variableList)/2)==0:
            gridTuple = (int(len(variableList)/ 2) + 1, 2)
        else:
            gridTuple = (int(len(variableList) / 2) , 2)
        az.plot_posterior(dataDict, var_names=variableList, hdi_prob=hdi_prob,grid=gridTuple)
        plt.savefig("plotPosterior.jpg")
        plt.savefig("plotPosterior.eps")
        plt.show()

    @classmethod
    def plotAutoCorr(cls,variableList):
        """
        Bar plot of the autocorrelation function for a sequence of data
        Rxx(tau)=x(t)x(t+tau)dt,from -inf to inf integration
        """
        returnSample=ArvizPlot.loadData()
        returnArray = np.array(returnSample)
        dataDict = {}
        for i1 in range(len(variableList)):
            dataDict[variableList[i1]] = returnArray[:, i1]
        az.style.use("arviz-whitegrid")
        if int(len(variableList)/2)==0:
            gridTuple = (int(len(variableList)/ 2) + 1, 2)
        else:
            gridTuple = (int(len(variableList) / 2) , 2)
        az.plot_autocorr(dataDict, var_names=variableList,grid=gridTuple)
        plt.savefig("plotAutoCorr.jpg")
        plt.savefig("plotAutoCorr.eps")
        plt.show()

    @classmethod
    def summary(cls,variableList):
        """
        Create a data frame with summary statistics.
        """
        returnArray=ArvizPlot.loadData()
        dataDict = {}
        for i1 in range(len(variableList)):
            dataDict[variableList[i1]] = [returnArray[:, i1],returnArray[:, i1]]
        az.style.use("arviz-whitegrid")
        if int(len(variableList) / 2) == 0:
            gridTuple = (int(len(variableList) / 2) + 1, 2)
        else:
            gridTuple = (int(len(variableList) / 2), 2)
        azSummary=az.summary(dataDict,var_names=variableList)
        print(azSummary)
#########################################################################
#########################################################################
class SqliteDB(object):
    """Save the data to sqlite database"""
    def __init__(self,dbPath):
        """
        Initialize the class
        Inputs:
            dbPath(str)-the path of the database
        """
        self._dbPath = dbPath

    @classmethod
    def initDB(self,dbPath):
        """Initialize the database"""
        self.db = records.Database('sqlite:///' + dbPath)  # connect to the sqlite database
        tableNames=self.db.get_table_names() # get table names
        for each in tableNames:
            self.db.query("DROP TABLE IF EXISTS " + each)  # delete all existing tables

    def saveSamples(self,sampleList):
        """Save samples to database, [[tag1,para1,para2,...],[],...]"""
        db = records.Database('sqlite:///' + self._dbPath)  # connect to the sqlite database
        samplesDict = [{'samples': str(each)} for each in sampleList]
        samplesTable = """
                        CREATE TABLE IF NOT EXISTS
                        samples(
                        samples MESSAGE_TEXT NOT NULL);"""
        db.query(samplesTable)  # create tables
        insertSamples = """
                        INSERT INTO
                        samples(samples)
                        values (:samples)  
                            """  # items in values are real parameters
        db.bulk_query(insertSamples, samplesDict)

    def getSamples(self):
        """return samples from database """
        db = records.Database('sqlite:///' + self._dbPath)
        conn = db.get_connection()  # connect to the sqlite database
        try:
            queryValue = conn.query('select * from samples;')
            returnValue = queryValue.all(as_dict=True)  # return samples as dict list
            return returnValue
        except:
            print("table samples doesn't exitst!")
            return

    @staticmethod
    def strToListConvert(strList,keys,types):
        """
        convert str list to numerical list，'[1,2,3]'to [1,2,3]
        Inputs:
            strList(dict list)
            keys(str)-the key of the dict
            types(str)-the target type ("int","float","list")
            eval() function convert str expression to its value
        """
        temp1=[eval(each[keys]) for each in strList]
        if types=='float':
            returnList = [list(map(float, each)) for each in temp1]
            finalList=returnList
        elif types=='int':
            returnList = [list(map(float, each)) for each in temp1]
            finalList=[list(map(eval(types),each)) for each in returnList]
        elif types=='list':
            finalList={each['tags']:eval(each[keys]) for each in strList}
        else:
            print("types must be float or int!")
        return finalList
#########################################################################
#########################################################################
class PyNUTS():
    """
    This class implements the No-U-Turn Sampler(NUTS) proposed by Hoffman and Gelman(The No-U-Turn Sampler:
    Adaptively Setting PathLengths in Hamiltonian Monte Carlo,Journal of Machine Learning Reserach 15(2014),1593-1623)
    """
    def __init__(self,numSampling,numBurning,acceptRatio):
        """
        Initialize the base class
        Inputs:
            numSampling(int)-total number of Sampling
            numBurning(int)-the number of steps excluded at the start of the sampling chain
            acceptRatio(float)-expected sampling accept ratio (0,1)
        """
        self.numSampling,self.numBurning = numSampling,numBurning
        self.acceptRatio=acceptRatio

    def leapFrog(self,theta,r,epsilon,logpGrad,pdFunc):
        """
        Do a leapfrog in Hamiltonian space
        Inputs:
            theta(float,array)-parameter values
            r(float,array)-momentum values
            epsilon(float)-step size
            logpGrad(float,array)-the gradient value of pdFunc at current parameter position
            pdFunc(callable probabilistic density function)
        Outputs:
            thetaWave(float,array)-the updated parameter values
            rWave(float,array)-the updated momentum values
            logpWave(float)-the updated value of Func
            logpGradWave(float,array)-the updated gradient value of pdFunc at updated parameter position
        """
        rWave=r+0.5*epsilon*logpGrad
        thetaWave=theta+epsilon*rWave
        logpWave, logpGradWave = pdFunc(thetaWave)
        rWave=rWave+0.5*epsilon*logpGradWave
        return thetaWave,rWave,logpWave,logpGradWave

    def findReasonableEpsilon(self,theta0,logpGrad0,logp0,pdFunc):
        """
        Heuristic for choosing an initial value of epsilon
        Inputs:
            theta0(float,array)-parameter values
            logpGrad0(float,array)-the gradient value of pdFunc at current parameter position
            logp0(float)-the value of Func
            pdFunc(callable probabilistic density function)
        Outputs:
            epsilon(float)-the estimated step size
        """
        epsilon=1
        r=np.random.normal(0,1,len(theta0))
        thetaDot,rDot,logpDot,logpGradDot=self.leapFrog(theta0, r, epsilon,logpGrad0, pdFunc)
        ###p(theta,r)~exp(l(theta)-0.5*r*r),l(theta)-the logarithm of the joint density of the varibles of interest theta
        ###logAcceptProb=log(p(thetaDot,rDot))-log(p(theta,r))
        logAcceptProb=logpDot-logp0-0.5*(np.dot(rDot,rDot)-np.dot(r,r))
        a=1 if logAcceptProb>np.log(0.5) else -1##2*I(logAcceptProb>0.5)-1
        while(a*logAcceptProb>(-a*np.log(2))):##((p(thetaDot,rDot)/p(theta,r))**a>2**(-a)
            epsilon=epsilon*(2.0**a)
            theta2Dot, r2Dot, logp2Dot, logp2GradDot = self.leapFrog(theta0, r, epsilon, logpGrad0, pdFunc)
            logAcceptProb=logp2Dot-logp0-0.5*(np.dot(r2Dot,r2Dot)-np.dot(r,r))
        return epsilon

    def buildTree(self,theta,r,logp,logpGrad,logu,vj,j,epsilon,pdFunc,lnPthetaR0):
        """
        Build the binary tree
        Inputs:
        Outputs:
        """
        if(j==0):
            ###Base case|take one leapfrog step in the direction vj.
            thetaDot,rDot,logpDot,logpGradDot=self.leapFrog(theta,r,vj*epsilon,logpGrad,pdFunc)
            logPthetaR=logpDot-0.5*np.dot(rDot,rDot)
            nDot=1 if logu<=logPthetaR else 0
            sDot=1 if logu<(logPthetaR+1000.0) else 0###deltaMax=1000
            deltalogPorb=logPthetaR-lnPthetaR0
            tempAlpha=min(1,np.exp(deltalogPorb))
            return (thetaDot,rDot,logpDot,logpGradDot,thetaDot,rDot,logpDot,logpGradDot,thetaDot,logpDot,logpGradDot,
                    nDot,sDot,tempAlpha,1)
        else:
            ###Recursion|implicitly build the left and right subtrees.
            elseReturnValues=self.buildTree(theta, r, logp, logpGrad, logu, vj, j-1, epsilon, pdFunc,lnPthetaR0)
            thetaMinus=elseReturnValues[0]
            rMinus = elseReturnValues[1]
            logpMinus = elseReturnValues[2]
            logpGradMinus = elseReturnValues[3]
            thetaPlus = elseReturnValues[4]
            rPlus = elseReturnValues[5]
            logpPlus = elseReturnValues[6]
            logpGradPlus = elseReturnValues[7]
            thetaDot=elseReturnValues[8]
            logpDot=elseReturnValues[9]
            logpGradDot=elseReturnValues[10]
            nDot=elseReturnValues[11]
            sDot=elseReturnValues[12]
            alphaDot=elseReturnValues[13]
            nalphaDot=elseReturnValues[14]
            if(sDot==1):
                if(vj==-1):
                    elseMinusReturnValues = self.buildTree(thetaMinus, rMinus, logpMinus, logpGradMinus,
                                                           logu, vj, j - 1, epsilon, pdFunc,lnPthetaR0)
                    thetaMinus=elseMinusReturnValues[0]
                    rMinus = elseMinusReturnValues[1]
                    logpMinus = elseMinusReturnValues[2]
                    logpGradMinus = elseMinusReturnValues[3]
                    theta2Dot = elseMinusReturnValues[8]
                    logp2Dot = elseMinusReturnValues[9]
                    logpGrad2Dot = elseMinusReturnValues[10]
                    n2Dot = elseMinusReturnValues[11]
                    s2Dot = elseMinusReturnValues[12]
                    alpha2Dot = elseMinusReturnValues[13]
                    nalpha2Dot = elseMinusReturnValues[14]
                else:
                    elsePlusReturnValues = self.buildTree(thetaPlus, rPlus, logpPlus, logpGradPlus,
                                                           logu, vj, j - 1, epsilon, pdFunc,lnPthetaR0)
                    thetaPlus = elsePlusReturnValues[4]
                    rPlus = elsePlusReturnValues[5]
                    logpPlus = elsePlusReturnValues[6]
                    logpGradPlus = elsePlusReturnValues[7]
                    theta2Dot = elsePlusReturnValues[8]
                    logp2Dot = elsePlusReturnValues[9]
                    logpGrad2Dot = elsePlusReturnValues[10]
                    n2Dot = elsePlusReturnValues[11]
                    s2Dot = elsePlusReturnValues[12]
                    alpha2Dot = elsePlusReturnValues[13]
                    nalpha2Dot = elsePlusReturnValues[14]
                acceptProbValue=float(n2Dot)/(max(1,float(nDot+n2Dot)))###n2Dot/(nDot+n2Dot),in case of 0/0,
                if(np.random.uniform()<acceptProbValue):
                    thetaDot=theta2Dot
                    logpDot=logp2Dot
                    logpGradDot=logpGrad2Dot
                alphaDot=alphaDot+alpha2Dot
                nalphaDot=nalphaDot+nalpha2Dot
                minusIndicateValue=np.dot((thetaPlus-thetaMinus),rMinus)
                plusIndicateValue=np.dot((thetaPlus-thetaMinus),rPlus)
                indValue=1 if (minusIndicateValue>=0)and(plusIndicateValue>=0) else 0
                sDot=indValue*s2Dot
                nDot=nDot+n2Dot
            return(thetaMinus,rMinus,logpMinus,logpGradMinus,thetaPlus,rPlus,logpPlus,logpGradPlus,thetaDot,
                   logpDot,logpGradDot,nDot,sDot,alphaDot,nalphaDot)

    def nuts(self,pdFunc,theta0):
        """
        Implements the No-u-Turn Sampler(NUTS) algorithm 6
        Inputs:
            pdFunc(callable function)-the probabilistic density function of the solved function
                logp,logpGrad=pdFunc(initParas)
            theta0(float,array)-initial values of the sample parametrs
        Outputs:

        """
        if len(np.shape(theta0))>1:
            raise ValueError('the initial parameter values should be 1d array')
        numPara=len(theta0)##the number of random variables
        sampleArray=np.zeros((self.numSampling+self.numBurning,numPara),dtype=float)
        logpArray=np.zeros((self.numSampling+self.numBurning,1),dtype=float)
        logp,logpGrad=pdFunc(theta0)
        sampleArray[0,:]=theta0
        logpArray[0,0]=logp
        ##Heuristic for choosing an initial value of epsilon
        epsilon=self.findReasonableEpsilon(theta0, logpGrad,logp, pdFunc)
        ###const parameters Setting
        miu=np.log(10.0*epsilon)
        gamma=0.05
        t0=10
        ka=0.75
        epsilonBar=1.0
        HBar=0.0;
        for m in trange(1,self.numSampling+self.numBurning):
            r0=np.random.normal(0,1,numPara)
            lnPthetaR0=logp-0.5*np.dot(r0,r0)
            ###u~uniform([0,exp(x)]),equals to (logu-x)~exponential(1)
            ###each distribution can be transformed to [0,1] uniform distribution, vice versa
            ###Fx(a)=p(x<=a),P(Fx(X)<=a)=P(X<=invFx(a))=Fx(invFx)=a, so Y=Fx(X)~U[0,1]
            pthetaR=logp-0.5*np.dot(r0,r0)
            logu=float(pthetaR-np.random.exponential(1,size=1))
            sampleArray[m,:]=sampleArray[m-1,:]
            logpArray[m,0]=logpArray[m-1,0]
            thetaMinus=sampleArray[m-1,:]
            thetaPlus=sampleArray[m-1,:]
            rMinus=r0
            rPlus=r0
            logpGradMinus=logpGrad
            logpGradPlus=logpGrad
            logpMinus=logp
            logpPlus=logp

            j,n,s=0,1,1
            while(s==1):
                vj=np.random.choice([-1,1],size=1,p=[0.5,0.5])[0]##vj~U([-1,1])
                if(vj==-1):
                    minusReturnValues=self.buildTree(thetaMinus,rMinus,logpMinus,logpGradMinus,logu,vj,j,epsilon,pdFunc,lnPthetaR0)
                    thetaMinus=minusReturnValues[0]
                    rMinus=minusReturnValues[1]
                    logpMinus=minusReturnValues[2]
                    logpGradMinus=minusReturnValues[3]
                    thetaDot=minusReturnValues[8]
                    logpDot=minusReturnValues[9]
                    logpGradDot=minusReturnValues[10]
                    nDot=minusReturnValues[11]
                    sDot=minusReturnValues[12]
                    alpha=minusReturnValues[13]
                    nalpha=minusReturnValues[14]
                else:
                    plusReturnValues = self.buildTree(thetaPlus, rPlus, logpPlus, logpGradPlus, logu, vj, j,
                                                       epsilon, pdFunc, lnPthetaR0)
                    thetaPlus = plusReturnValues[4]
                    rPlus = plusReturnValues[5]
                    logpPlus = plusReturnValues[6]
                    logpGradPlus = plusReturnValues[7]
                    thetaDot = plusReturnValues[8]
                    logpDot = plusReturnValues[9]
                    logpGradDot = plusReturnValues[10]
                    nDot = plusReturnValues[11]
                    sDot = plusReturnValues[12]
                    alpha = plusReturnValues[13]
                    nalpha = plusReturnValues[14]
                if (sDot==1):
                    acceptProbs=min(1,float(nDot)/float(n))
                    if (np.random.uniform()<acceptProbs):
                        sampleArray[m,:]=thetaDot
                        logpArray[m,0]=logpDot
                        logp=logpDot
                        logpGrad=logpGradDot
                n=n+nDot
                minusIndicateValue = np.dot((thetaPlus - thetaMinus), rMinus)
                plusIndicateValue = np.dot((thetaPlus - thetaMinus), rPlus)
                indValue = 1 if (minusIndicateValue >= 0) and (plusIndicateValue >= 0) else 0
                s=sDot*indValue
                j=j+1
            if(m<=self.numBurning):
                temp1=1.0/float(m+t0)
                HBar=(1.0-temp1)*HBar+temp1*(self.acceptRatio-float(alpha)/float(nalpha))
                epsilon=np.exp(miu-np.sqrt(m)/gamma*HBar)
                temp2=m**(-ka)
                epsilonBar=np.exp(temp2*np.log(epsilon)+(1.0-temp2)*np.log(epsilonBar))
            else:
                epsilon=epsilonBar
        return sampleArray[self.numBurning:,:],logpArray[self.numBurning:,:],epsilon

#########################################################################
#########################################################################
class MultiNormal(PyNUTS):
    """
    A class for multivariate normal distribution sampling
    """
    def __init__(self,meanArray, covArray, numSampling, numBurning, initalThetaArray,acceptRatio=0.65):
        """
        Initialize the data for the class
            Inputs:
                meanArray(float,array)-mean array, eg. [0,0]
                covArray(float,array)-covariance array, eg. [[1,1.98],[1.98,4]]
                numSampling(int)-total number of numSampling
                numBurning(int)-the number of steps excluded at the start of the sampling chain
                initialThetaArray(float,array)-initial values of the sample parametrs
                acceptRatio(float)-expected sampling accept ratio (0,1)
        """
        assert isinstance(meanArray,np.ndarray)
        assert isinstance(covArray,np.ndarray)
        assert isinstance(numSampling,int)and(numSampling>0)##make sure valid input values
        assert isinstance(numBurning,int)and(numBurning>0)
        assert isinstance(initalThetaArray,np.ndarray)
        assert isinstance(acceptRatio,float)and(0<acceptRatio<1)
        super().__init__(numSampling,numBurning,acceptRatio)
        self.meanArray =meanArray
        self.covArray=covArray
        self.initialThetaArray=initalThetaArray

    def sample(self):
        """
        The method for sampling
        """
        sampleArray,logpArray,epsilon=self.nuts(self.multiNormalFunc,self.initialThetaArray)###run the method in base class
        return sampleArray

    def multiNormalFunc(self,thetaArray):
        """
        A multivariate_normal probabilistic density function sampling
        f(x)=exp(-0.5*(x-mean)Tsigma-1(x-mean) exclude the constant coefficient
        Inputs:
            thetaArray(float,array)-the parameters values
        Outputs:
            logp(float)-the natural logarithm of probability density function
            logpGrad(float,array)-the gradient of logp
        """
        thetaArray=thetaArray.astype(np.float)
        invCoxArray=np.linalg.inv(self.covArray)
        x=torch.tensor(thetaArray,requires_grad=True)#tranform numpy array to torch tensor
        torchMean=torch.tensor(self.meanArray)
        torchInvcOV=torch.tensor(invCoxArray)
        torchTemp=torch.mm((x-torchMean),torchInvcOV)
        y=-0.5*torch.mm(torchTemp,(x-torchMean).T)
        y.backward() #calculate the derivative of y with respect to x
        logpGrad=x.grad.numpy()
        logp=y.detach().numpy()[0,0]##transform a tensor that requres grad into numpy
        return logp,logpGrad
#########################################################################
##################---generalized linear regression---####################
class LinearRegression(PyNUTS):
    """
    A generalized linear regression class
    """
    def __init__(self, inputList, outputList,numSampling, numBurning,acceptRatio=0.65):
        """
        Initialize the data for the class
        """
        assert isinstance(numSampling, int) and (numSampling > 0)  ##make sure valid input values
        assert isinstance(numBurning, int) and (numBurning > 0)
        assert isinstance(acceptRatio, float) and (0 < acceptRatio < 1)
        super().__init__(numSampling, numBurning, acceptRatio)
        self.numRow,self.numCol=np.shape(inputList)[0],np.shape(inputList)[1]
        inputArray=np.array(inputList)
        self.xMax=np.max(inputArray,axis=0)
        self.xMin=np.min(inputArray,axis=0)
        outputArray=np.array([outputList]).T
        self.yMax=np.max(outputArray,axis=0)
        self.yMin=np.min(outputArray,axis=0)
        xArrayNormalize=self._normalize(inputArray)
        yArrayNormalize=self._normalize(outputArray)
        self.inputArray=xArrayNormalize
        self.outputArray=yArrayNormalize
        initalThetaArray=[]
        for i1 in range(self.numCol+1):
            sampleValue=np.random.normal(0,1)
            initalThetaArray.append(sampleValue)
        initalThetaArray[-1]=np.abs(initalThetaArray[-1])
        self.initialThetaArray = np.array(initalThetaArray)

    def _normalize(self,inputArray):
        """
        ---Inout values normalize---
        Inputs:inputArray(float array)-f.g. [[1,0.3,0.4],[1,0.5,0.7],...]
        Outputs:normalized inputArray (x-xmin)/(xmax-xmin)
        """
        if np.shape(inputArray)[1]>1:###for inputArray
            deltaX=self.xMax-self.xMin
            for i1 in range(1,self.numCol):
                inputArray[:,i1]=(inputArray[:,i1]-self.xMin[i1])/float(deltaX[i1])
        else:##for outputArray
            inputArray=(inputArray-self.yMin)/float(self.yMax-self.yMin)
        return inputArray

    def _returnOrign(self,sampleArray):
        """
        ---Recover the sample parameters---
        Inputs:sampleArray(float,array)-the samples corresponding to the normalized observations
        Outputs:recover the sample parameters
        """
        deltay=self.yMax-self.yMin
        interceptTerm1=np.zeros((np.shape(sampleArray)[0],1))
        for i1 in range(1,self.numCol):
            interceptTerm1[:,0]-=sampleArray[:,i1]*deltay*self.xMin[i1]/float(self.xMax[i1]-self.xMin[i1])
        sampleArray[:,0]=interceptTerm1[:,0]+sampleArray[:,0]*deltay+self.yMin##intercept term
        sampleArray[:,-1]=(sampleArray[:,-1]*deltay)**2##deviation term
        for i2 in range(1,self.numCol):
            sampleArray[:,i2]=sampleArray[:,i2]*deltay/float(self.xMax[i2]-self.xMin[i2]) ## middle term
        return sampleArray

    def probFunc(self,thetaArray):
        """
        A linear regression probabilistic density function sampling
        f(x)=1/((2pi)**0.5*sigma)exp(-(Y-wx)T(Y-wx)/(2*sigma**2) exclude the constant coefficient
        Inputs:
            thetaArray(double,array)-the parameters values
        Outputs:
            logp(double)-the natural logarithm of probability density function
            logpGrad(double,array)-the gradient of logp
        """
        num=np.shape(thetaArray)[0]
        numInput=np.shape(self.inputArray)[0]
        # thetaTorch=torch.tensor(thetaArray,requires_grad=True)#tranform numpy array to torch tensor
        xTorch=torch.tensor(self.inputArray)
        yTorch=torch.tensor(self.outputArray)
        thetaTorch1=np.zeros((num-1,1))
        thetaTorch1[:,0]=thetaArray[:-1]
        thetaTorch2=(thetaArray[-1])**2
        thetaTorchX=torch.tensor(thetaTorch1,requires_grad=True)
        thetaTorchSigma2=torch.tensor(thetaTorch2,requires_grad=True)
        torchTemp1=yTorch-torch.mm(xTorch,thetaTorchX)
        y=(-0.5*torch.mm(torchTemp1.T,torchTemp1))/thetaTorchSigma2-0.5*numInput*torch.log(thetaTorchSigma2)\
          -0.5*numInput*np.log(2.0*np.pi)##logp=-0.5*(x-mu)T(x-mu)/sigma2-0.5*n*log(sigma2)-0.5*n*log(2pi)
        y.backward()
        logpGrad1=thetaTorchX.grad.numpy()
        logpGrad2=thetaTorchSigma2.grad.numpy()
        logpGrad=[]
        for i in range(num-1):
            logpGrad.append(logpGrad1[i][0])
        logpGrad.append((logpGrad2).tolist()*2.0*thetaArray[-1])
        logp=y.detach().numpy()[0,0]##transform a tensor that requres grad into numpy
        return logp,np.array(logpGrad)

    def sample(self):
        """
        The method for sampling
        """
        sampleArray,logpArray,epsilon=self.nuts(self.probFunc,self.initialThetaArray)###run the method in base class
        sampleArray=self._returnOrign(sampleArray)
        sampleList=[list(sampleArray[i1,:]) for i1 in range(np.shape(sampleArray)[0])]
        dbPath = "samplesDB.db"
        saveInstance = SqliteDB(dbPath)
        SqliteDB.initDB(dbPath)
        saveInstance.saveSamples(sampleList)


        # samples = sampleArray[1::10, :]
        # print('Percentiles')
        # print(np.percentile(samples, [16, 50, 84], axis=0))
        # print('Mean')
        # print(np.mean(samples, axis=0))
        # print('Stddev')
        # print(np.std(samples, axis=0))

    def plotTrace(self):
        """
        Plot the trace of the samples
        Inputs:None
        """
        variaNum=np.shape(self.inputArray)[1]
        variableList=["beta_"+str(i1) for i1 in range(variaNum)]
        variableList.append("sigma2")
        plotTrace=ArvizPlot.plotTrace(variableList)

    def plotPosterior(self,hdi_prob=0.95):
        """
        Plot the posterior distribution of the parameter
        Inputs:
            hdi_prob(float)-Plots highest density interval for chosen percentage of density.
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        variableList.append("sigma2")
        ArvizPlot.plotPosterior(variableList)

    def plotAutoCorr(self):
        """
        Bar plot of the autocorrelation function for a sequence of data
        Rxx(tau)=x(t)x(t+tau)dt,from -inf to inf integration
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        variableList.append("sigma2")
        ArvizPlot.plotAutoCorr(variableList)

    def summary(self):
        """
        Create a data frame with summary statistics.
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        variableList.append("sigma2")
        ArvizPlot.summary(variableList)
#########################################################################
########################---logistic regression---########################
class LogisticRegression(PyNUTS):
    """
     A logistic regression class
    """
    def __init__(self,inputList, outputList,numSampling, numBurning,acceptRatio=0.65):
        """Initialize the data for the class"""
        assert isinstance(numSampling, int) and (numSampling > 0)  ##make sure valid input values
        assert isinstance(numBurning, int) and (numBurning > 0)
        assert isinstance(acceptRatio, float) and (0 < acceptRatio < 1)
        super().__init__(numSampling, numBurning, acceptRatio)
        self.numRow, self.numCol = np.shape(inputList)[0], np.shape(inputList)[1]
        inputArray = np.array(inputList)
        self.xMax = np.max(inputArray, axis=0)
        self.xMin = np.min(inputArray, axis=0)
        outputArray = np.array([outputList]).T
        self.yMax = np.max(outputArray, axis=0)
        self.yMin = np.min(outputArray, axis=0)
        self.inputArray=inputArray
        self.outputArray=outputArray
        initalThetaArray = []
        for i1 in range(self.numCol):
            sampleValue = np.random.normal(0, 10)
            initalThetaArray.append(sampleValue)
        self.initialThetaArray = np.array(initalThetaArray)

    def probFunc(self, thetaArray):
        """
        :param thetaArray:
        :return:
        """
        num = np.shape(thetaArray)[0]
        numInput=np.shape(self.inputArray)[0]
        thetaArray =np.reshape(thetaArray,(num,1))
        inputArray =self.inputArray
        outputArray =self.outputArray
        E=np.ones(numInput)
        EArray=np.reshape(E,(numInput,1))
        logpValue=np.dot(np.dot(outputArray.T,inputArray),thetaArray)-np.dot(EArray.T,np.log(EArray+np.exp(np.dot(inputArray,thetaArray))))
        logpGradValue=np.dot(inputArray.T,(outputArray-1.0/(1.0+np.exp(-np.dot(inputArray,thetaArray)))))
        logpGrad = []
        for i in range(num):
            logpGrad.append(logpGradValue[i,0])
        logp=logpValue[0,0]
        # print(logp,logpGrad)
        return logp, np.array(logpGrad)

    def sample(self):
        """
        The method for sampling
        """
        sampleArray,logpArray,epsilon=self.nuts(self.probFunc,self.initialThetaArray)###run the method in base class
        dbPath = "samplesDB.db"
        saveInstance = SqliteDB(dbPath)
        SqliteDB.initDB(dbPath)
        saveInstance.saveSamples(sampleArray.tolist())

    def plotTrace(self):
        """
        Plot the trace of the samples
        Inputs:None
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        ArvizPlot.plotTrace(variableList)

    def plotPosterior(self, hdi_prob=0.95):
        """
        Plot the posterior distribution of the parameter
        Inputs:
            hdi_prob(float)-Plots highest density interval for chosen percentage of density.
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        ArvizPlot.plotPosterior(variableList)

    def plotAutoCorr(self):
        """
        Bar plot of the autocorrelation function for a sequence of data
        Rxx(tau)=x(t)x(t+tau)dt,from -inf to inf integration
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        ArvizPlot.plotAutoCorr(variableList)

    def summary(self):
        """
        Create a data frame with summary statistics.
        """
        variaNum = np.shape(self.inputArray)[1]
        variableList = ["beta_" + str(i1) for i1 in range(variaNum)]
        ArvizPlot.summary(variableList)
#########################################################################
# if __name__ == '__main__':
#     #########################---multiNormal example---##########################
#     meanArray = np.array([[1.2,-3.4]])
#     covArray = np.array([[2.1,-1.2],[-1.2,5.2]])
#     numSample = 5000
#     numBurning = 5000
#     initalThetaArray = np.random.normal(0, 1, 2)
#     testInstance=MultiNormal(meanArray, covArray, numSample, numBurning, initalThetaArray)
#     testInstance.multiNormalFunc(initalThetaArray)
#     nutsSamples=testInstance.sample()
#     print('Mean')
#     print(np.mean(nutsSamples, axis=0))
#     print('Stddev')
#     print(np.std(nutsSamples, axis=0))
#     sampleValues = np.random.multivariate_normal(meanArray[0, :],covArray, size=numSample)
#     plt.plot(sampleValues[:, 0], sampleValues[:, 1], 'b.')
#     plt.plot(nutsSamples[:, 0], nutsSamples[:, 1], 'r+')
#     plt.savefig("multivariateSample.jpg")
#     plt.savefig("multivariateSample.eps")
#     plt.show()
#     plt.subplot(2, 2, 1)
#     plt.hist(nutsSamples[:, 0], bins=50,color='red')
#     plt.xlabel("x1-nutsSamples")
#     plt.subplot(2, 2, 2)
#     plt.hist(sampleValues[:, 0], bins=50, color='blue')
#     plt.xlabel("x1-npSamples")
#     plt.subplot(2, 2, 3)
#     plt.hist(nutsSamples[:, 1], bins=50,color='red')
#     plt.xlabel("x2-nutsSamples")
#     plt.subplot(2, 2, 4)
#     plt.hist(sampleValues[:, 1], bins=50, color='blue')
#     plt.xlabel("x2-npSamples")
#     plt.savefig("multiNormalCompare.jpg")
#     plt.savefig("multiNormalCompare.eps")
#     plt.show()

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
    ##############---logistic regression example---############
    # xy=np.loadtxt("logisticRegression.txt")
    # x=list(xy[:,0])
    # y=list(xy[:,1])
    # x=[[1.0,np.log(each)] for each in x]
    # print(x)
    # plt.plot(x,y,"*")
    # plt.show()
    # print(1.0/(1.0+np.exp(-0.593)))
    #
    # numSampling=20000
    # numBurning=5000
    # logisticInstance=LogisticRegression(x,y,numSampling,numBurning)
    # logisticInstance.sample()
    # logisticInstance.plotTrace()
    # logisticInstance.plotPosterior()
    # logisticInstance.plotAutoCorr()
    # logisticInstance.summary()




    ######################################################################








