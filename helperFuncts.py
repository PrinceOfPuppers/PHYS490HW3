import numpy as np
import matplotlib.pyplot as plt

def stateToStateNumber(state,stateSize):
    #stateNumber is a representation 
    # of the state in base 10
    #ie -1 1 1 -1 = 0110 = 8*0 + 4*1 +2*1 1*0 = 6
    stateNumber=0

    for i in range(0,stateSize):
        value=2^i
        index=stateSize-i-1

        binaryVal=(state[index]+1)/2

        stateNumber+=value*binaryVal
    
    return(int(stateNumber))

def stateNumberToState(stateNumber,stateSize):
    state=np.zeros((stateSize,1))
    binary=format(stateNumber,"b")
    binary= (stateSize-len(binary))*"0"+binary
    
    for i,digit in enumerate(binary):
        state[i]= float(2*(int(digit)-1/2))
        
    return(state)

def saveDictToTxt(myDict,outputPath):
    dictStr=str(myDict)
    outData=open(outputPath,"w+")
    outData.write(dictStr+"\n")

def graphList(myList,title,xlabel,ylabel):
    fig=plt.plot(myList)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()