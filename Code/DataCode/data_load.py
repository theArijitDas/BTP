'''
data generator
+ datagenerator(seed, maximum number of files)
+ generating at every time instant a random data(): what probability distribution to use - simple
+ extreme random
'''
import random
import os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class RandomDataGenerator:
    def __init__(self, N: int, distribution: str="uniform", seed: int=42,oneHotEncoded: bool=False):
        seed_everything(seed)
        self.distribution=distribution.lower()
        self.N=N
        self.oneHotEncoded=oneHotEncoded
    def getOneHotEncoded(self, a:int):
        '''
        generates one hot encoded vector(numpy array)
        '''
        zeros_array = np.zeros(self.N, dtype=np.int32)
        zeros_array[a]=1
        return zeros_array
    
    def generateFromDistribution(self):
        '''
            generates a pair of number from the given distribution
        '''
        if(self.distribution=="uniform"):return np.random.randint(0, self.N, size=2)
        if(self.distribution=="normal"):
            arr=np.random.normal(loc=self.N/2, scale=self.N/6, size=2).astype(np.int32)
            i=1
            while i>=0:
                if(arr[i]>=self.N):arr[i]=self.N-1
                elif arr[i]<0: arr[i]=0
                i-=1
            return arr


    def generate(self):
        '''
        returns list of one hot encoded vectors if one hot encoded is true
        returns list of size 2 of integers 
        '''
        arr=self.generateFromDistribution()
        if(self.oneHotEncoded):
            return [self.getOneHotEncoded(arr[0]),self.getOneHotEncoded(arr[1])]
        return arr