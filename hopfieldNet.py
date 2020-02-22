import numpy as np

def heaviside(vec,threshold=0):
    vec=np.heaviside(vec,0)
    vec=vec-1/2
    vec=2*vec
    return(vec)

class Hfn:
    def __init__(self,numNodes,initalVals):
        self.numNodes=numNodes
        self.nodeVals=initalVals
        self.weights=np.zeros((numNodes,numNodes))
        self.thresholds=np.zeros((numNodes,1))
    
    def synchUpdate(self):
        self.nodeVals=heaviside(np.dot(self.weights,self.nodeVals),self.thresholds)
    
    def findMinima(self,state,maxIter):
        self.nodeVals=state
        prevVals=np.zeros((self.numNodes,1))
        self.synchUpdate()
        for i in range(0,maxIter):
            np.copyto(prevVals,self.nodeVals)
            self.synchUpdate()
            if np.array_equal(self.nodeVals,prevVals):
                break

        return(self.nodeVals)
    
    def hebbianLearning(self,state,numStates):
        self.weights+=(np.outer(state,state)-np.identity(self.numNodes))/numStates

    def getAdjacentCoupling(self):
        couplingDict=dict.fromkeys([(i,(i+1)%self.numNodes) for i in range(0,self.numNodes)])
        for i in range(0,self.numNodes):
            couplingDict[(i,(i+1)%self.numNodes)]=heaviside(self.weights[i,(i+1)%self.numNodes])

        return(couplingDict)