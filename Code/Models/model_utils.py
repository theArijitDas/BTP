'''
static optimum
regret calculation 
models - FTPL
'''
import numpy as np
class ModelUtils():
    
    def __init__(self,N: int,oneHotEncoded: bool=False):
        self.oneHotEncoded=oneHotEncoded
        self.N=N
        self.freqMatrix=np.zeros((N,N), dtype=np.int32) # reward matrix in presentation have discussion over the name before u change it
        self.maxValue=0
        self.maxPosition=[0,0]
        self.n=0 #number of time the reward matrix was updated or number of instances that have passed
        self.reward=0 # cumulative reward till this point
    
    def getIntegerFromOneHotEncoded(self,a: list[np.ndarray]):
        arr=[-1,-1]
        
        for d in range(2): # d stands for Dimension
            for i in range(len(a[d])):
                if(a[d][i]):
                    arr[d]=i
                    break
        return arr             

    def updatefreqMatrix(self,a):
        if(self.oneHotEncoded):a=self.getIntegerFromOneHotEncoded(a) # thiss all marked as "thiss" are expected to be list of two one hot encoded vectors
        self.freqMatrix[a[0],a[1]]+=1
        if(a[0]!=a[1]):
            self.freqMatrix[a[1],a[0]]+=1
        if(self.maxValue<=self.freqMatrix[a[0],a[1]]):
            self.maxValue=self.freqMatrix[a[0],a[1]]
            self.maxPosition=[a[0],a[1]]
        self.n+=1

    def getStaticOptimumReward(self): # refers to the reward corresponding to best possible choice 
        '''
        May be implemented incorrectly : first discuss then change
        '''
        return self.maxValue
    
    def updateReward(self,predicted,requested):
        if(self.oneHotEncoded):
            requested=self.getIntegerFromOneHotEncoded(requested) # thiss
            predicted=self.getIntegerFromOneHotEncoded(predicted) # thiss
        predicted.sort()
        requested.sort()
        self.reward+= (predicted[0]==requested[0] and predicted[1]==requested[1])

