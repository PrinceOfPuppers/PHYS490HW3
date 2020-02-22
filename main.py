import sys
import argparse as ap
import numpy as np
from math import log
from hopfieldNet import Hfn
from chainNet import CyclicChainNet as ccn
from helperFuncts import stateToStateNumber,stateNumberToState,saveDictToTxt,graphList
from config import Config


def loadData(charDict):
    rawDataPath=sys.argv[1]
    outputPath="outputs/"+(rawDataPath.split("/")[-1])
    rawData=open(rawDataPath)
    inData=[]
    for line in rawData:
        line=line.strip()
        if not len(line)==0:
            lineData=[]
            for char in line:
                lineData.append(charDict[char])
            inData.append(lineData)
    
    inData=np.array(inData)
    print(">>{} states loaded each of size {}".format(inData.shape[0],inData.shape[1]))
    return(inData,outputPath)

def getProbDistHfn(hfn,stateSize):
    stateTypes=2**stateSize

    stateTypeCounterHfn=[0]*stateTypes
    
    for i in range(0,stateTypes):
        state=stateNumberToState(i,stateSize)
        state=hfn.findMinima(state,100)
        stateNumber=stateToStateNumber(state,stateSize)
        stateTypeCounterHfn[stateNumber]+=1

    #normalizes
    for i in range(0,stateTypes):
        stateTypeCounterHfn[i]/=stateTypes
    
    return(stateTypeCounterHfn)


def getProbDistData(states):
    numStates=states.shape[0]
    stateSize=states.shape[1]
    
    stateTypes=2**stateSize
    stateTypeCounterData=[0]*stateTypes

    for state in states:
        stateNumber=stateToStateNumber(state,stateSize)
        stateTypeCounterData[stateNumber]+=1
    
    #normalizes
    for i in range(0,stateTypes):
        stateTypeCounterData[i]/=numStates
    
    return(stateTypeCounterData)

def getKLDivergence(probDist1,probDist2,stateSize):
    KLDiv=0
    for i in range(0,2**stateSize):
            if probDist2[i]!=0:
                if probDist1[i]/probDist2[i]!=0:
                    KLDiv+=probDist1[i]*log(probDist1[i]/probDist2[i])

    return(KLDiv)

def trainHfn(hfn,states,numStates,epochs,verbose):
    KLDivergence=[0]*epochs
    if verbose:
        prodDistData=getProbDistData(states)
    
    print("Training:")
    for i in range(0,epochs):

        for state in states:
            hfn.hebbianLearning(state,numStates)
        
        if verbose:
            probDistHfn=getProbDistHfn(hfn,states.shape[1])
            KLDivergence[i]=getKLDivergence(probDistHfn,prodDistData,states.shape[1])

    return(KLDivergence)

def main(verbose,cfg):
    charDict={"+": 1.0, "-": -1.0}

    states,outputPath=loadData(charDict)

    numStates=states.shape[0]
    stateSize=states.shape[1]

    hfn=Hfn(stateSize,np.ones((stateSize,1)))

    KLDivergence=trainHfn(hfn,states,numStates,cfg.epochs,verbose)


    couplingDict=hfn.getAdjacentCoupling()
    print(">>Coupling Dictionary saved to: {}".format(outputPath))
    saveDictToTxt(couplingDict,outputPath)

    if verbose:
        print(">>Hopfield Network Weights:")
        print(hfn.weights)

        print(">>Graphing KL Divergence")
        title="KL Divergence Of Hopfield Network and Training Dataset over {} Epochs".format(cfg.epochs)
        xLabel="Epoch"
        yLabel="KL Divergence"

        graphList(KLDivergence,title,xLabel,yLabel)


if __name__ == "__main__":
    cfg=Config()
    verbose=False
    try:
        if sys.argv[2] in ("-v","--v","-verbose", "--verbose"):
            verbose=True
    except:
        pass
    main(verbose,cfg)