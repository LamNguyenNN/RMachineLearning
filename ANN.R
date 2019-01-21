initWeightMats = function(topo) {
  i = 1
  weightMats = list()
  while(i < length(topo)) {
    weightMats[[i]] = matrix(data = rnorm(n=topo[i]*topo[i+1], mean=0, sd=1), nrow=topo[i], ncol=topo[i+1], byrow=T)
    i = i + 1
  }
  return (weightMats)
}

initBiasMats = function(topo, numInputSets) {
  i = 1
  biasMats = list()
  while(i < length(topo)) {
    biasMats[[i]] = matrix(data = rnorm(n=numInputSets*topo[i+1], mean=0, sd=1), nrow=numInputSets, ncol=topo[i+1], byrow=T)
    i = i + 1
  }
  return (biasMats)
}

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

elu = function(x) {
  ifelse(x > 0, x, eluAlpha * (exp(x) - 1))
}
eluDerivative = function(x) {
  ifelse(x > 0, 1, eluAlpha * exp(x) )
}

softmax = function(x) {
  sum = sum(exp(x))
  exp(x)/sum
}

forwardProp = function(inputMat, weightMats, biasMats) {
  numSynapses = length(weightMats)
  
  activatedSums = list()
  
  activatedSums[[1]] = elu((inputMat %*% weightMats[[1]]) + biasMats[[1]])
  
  i = 2
  while(i < numSynapses) {
    activatedSums[[i]] = elu((activatedSums[[i-1]] %*% weightMats[[i]]) + biasMats[[i]])
    i = i + 1
  }
  
  output = (activatedSums[[numSynapses-1]] %*% weightMats[[numSynapses]]) 
                                       + biasMats[[numSynapses]]
  
  for(i in 1:nrow(output)) {
    output[i,] = softmax(output[i,])
  }
  
  return (list("activatedSums" = activatedSums, "output" = output))
}

MSECost = function(targetOutput, netOutput) {
  error = (1/nrow(targetOutput)) * sum( (1/ncol(targetOutput))*(rowSums((targetOutput - netOutput) ^ 2)) )
  return(error)
}

CrossEntropyCost = function(targetOutput, netOutput) {
  error = (1/nrow(targetOutput)) * sum( rowSums(-targetOutput * log(netOutput)))
  return(error)
}

SGD = function(inputMat, weightList, biasList, outputList, targetOutput) {
  
  synapseIndex = length(weightList)
  hiddenToOutputGrad = weightList[[synapseIndex]]
  
  for(trainEx in 1:nrow(outputList$output)) {
  
    deltaList = list()
    
    hiddenToOutputDelta = weightList[[synapseIndex]]
    
    for(i in 1:nrow(hiddenToOutputGrad)) {
      for(j in 1:ncol(hiddenToOutputGrad)) {
        delta = (outputList$output[trainEx,j] 
                - targetOutput[trainEx,j])
        hiddenToOutputDelta[i,j] = delta;
        hiddenToOutputGrad[i,j] = delta * outputList$activatedSums[[synapseIndex-1]][1, i]
      }
    }
    
    deltaList[[synapseIndex]] = hiddenToOutputDelta
    
    while(synapseIndex > 1) {
    
      synapseIndex = synapseIndex - 1
      hiddenDelta = weightList[[synapseIndex]]
      hiddenGrad = weightList[[synapseIndex]]
    
      for(i in 1:nrow(hiddenGrad)) {
        for(j in 1:ncol(hiddenGrad)) {
          delta = eluDerivative(outputList$activatedSums[[synapseIndex]][1, j]) * 
             sum(c(weightList[[synapseIndex+1]][j,]) * c(deltaList[[synapseIndex+1]][j,]))
          hiddenDelta[i,j] = delta
          if(synapseIndex == 1) {
            hiddenGrad[i,j] = delta * inputMat[1,i]
          } else {
            hiddenGrad[i,j] = delta * outputList$activatedSums[[synapseIndex-1]][1, i]
          }
        }
      }
      
      deltaList[[synapseIndex]] = hiddenDelta
      
    }
    
    synapseIndex = length(weightList)
    
  }
    
}

input = matrix(data = c(0,0, 1,0, 0,1, 1,1), nrow = 4, ncol = 2, byrow = T)
trainOutput = matrix(data = c(1,0, 0,1, 0,1, 1,0), nrow = 4, ncol = 2, byrow = T)

numInputSets = 4
numLayers = 3
eluAlpha = 1
topology = c(2,3,2)

weightList = initWeightMats(topology)
biasList = initBiasMats(topology, numInputSets)

print(weightList)
print(biasList)

outputList = forwardProp(input, weightList, biasList)
outputList

SGD(input, weightList, biasList, outputList, trainOutput)



