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
    biasMats[[i]] = matrix(data = rnorm(n=topo[i+1], mean=0, sd=1), nrow=numInputSets, ncol=topo[i+1], byrow=T)
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

SGD = function(inputMat, weightList, biasList, outputList, targetOutput, learningRate, epoch) {
  origInput_mat = inputMat
  synapseIndex = length(weightList)
  epochNum = 0
  while (epochNum < epoch) {
    for(trainEx in 1:nrow(targetOutput)) {
      outputList = forwardProp(inputMat, weightList, biasList)
      deltaWeightList = list()
      gradWeightList = list()
      gradBiasList = list() #Gradient is same as delta for bias
      
      deltaWeightList[[synapseIndex]] = weightList[[synapseIndex]]
      gradWeightList[[synapseIndex]] = matrix(nrow=nrow(weightList[[synapseIndex]]),
                                              ncol=ncol(weightList[[synapseIndex]]), byrow = T)
      gradBiasList[[synapseIndex]] = matrix(nrow=nrow(biasList[[synapseIndex]]), 
                                            ncol=ncol(biasList[[synapseIndex]]), byrow = T)
     
      
      #Hidden to Output gradient calculation
      for(i in 1:nrow( gradWeightList[[synapseIndex]])) {
        for(j in 1:ncol( gradWeightList[[synapseIndex]])) {
          delta = (outputList$output[trainEx,j] 
                   - targetOutput[trainEx,j])
          deltaWeightList[[synapseIndex]][i,j] = delta;
          gradWeightList[[synapseIndex]][i,j] = delta * outputList$activatedSums[[synapseIndex-1]][trainEx, i]
          
        }
      }
      
      for(i in 1:nrow( gradBiasList[[synapseIndex]])) {
        for(j in 1:ncol( gradBiasList[[synapseIndex]])) {
          delta = (outputList$output[trainEx,j] 
                   - targetOutput[trainEx,j])
          gradBiasList[[synapseIndex]][i, j] = delta;
        }
      }
      
      while(synapseIndex > 1) {
        synapseIndex = synapseIndex - 1
        deltaWeightList[[synapseIndex]] = weightList[[synapseIndex]]
        gradWeightList[[synapseIndex]] = weightList[[synapseIndex]]
        gradBiasList[[synapseIndex]] = matrix(nrow=nrow(biasList[[synapseIndex]]), 
                                              ncol=ncol(biasList[[synapseIndex]]), byrow = T)
        
        for(i in 1:nrow(gradWeightList[[synapseIndex]])) {
          for(j in 1:ncol(gradWeightList[[synapseIndex]])) {
            delta = eluDerivative(outputList$activatedSums[[synapseIndex]][trainEx, j]) * 
              sum(c(weightList[[synapseIndex+1]][j,]) * c(deltaWeightList[[synapseIndex+1]][j,]))
            deltaWeightList[[synapseIndex]][i,j] = delta
            if(synapseIndex == 1) {
              gradWeightList[[synapseIndex]][i,j] = delta * inputMat[trainEx,i]
            } else {
              gradWeightList[[synapseIndex]][i,j] = delta * outputList$activatedSums[[synapseIndex-1]][trainEx, i]
            }
          }
        }
        
        for(i in 1:nrow(gradBiasList[[synapseIndex]])) {
          for(j in 1:ncol(gradBiasList[[synapseIndex]])) {
            delta = eluDerivative(outputList$activatedSums[[synapseIndex]][trainEx, j]) * 
              sum(c(weightList[[synapseIndex+1]][j,]) * c(deltaWeightList[[synapseIndex+1]][j,]))
            gradBiasList[[synapseIndex]][i, j] = delta
          }
        }
      }
      
      synapseIndex = length(weightList)
      
      for(i in 1:synapseIndex) {
        weightList[[i]] = weightList[[i]] - (learningRate * gradWeightList[[i]])
        biasList[[i]] = biasList[[i]] - (learningRate * gradBiasList[[i]])
        #print(biasList[i])
      }
      
    }
    
   # newOutput = forwardProp(inputMat, weightList, biasList)
    #print(round(newOutput$output), digits = 3)
   
    epochNum = epochNum + 1
    print(epochNum) 
    randomSwap = sample(1:nrow(inputMat), nrow(inputMat), replace = F)
    
    inputMat = inputMat[randomSwap,]
    targetOutput = targetOutput[randomSwap,]
  }
  
  newBiasList = list()
  for(i in 1:length(biasList)) {
    newBiasList[[i]] = biasList[[i]][1,]
  }
  print(forwardProp(origInput_mat, weightList, biasList)$output)
  return (list("weights" = weightList, "biases" = newBiasList))
  
}


if(F) {
input = matrix(data = c(0,0, 1,0, 0,1, 1,1), nrow = 4, ncol = 2, byrow = T)
trainOutput = matrix(data = c(1,0, 0,1, 0,1, 1,0), nrow = 4, ncol = 2, byrow = T)

numInputSets = 4
numLayers = 3
eluAlpha = .7
learningRate = .25
epoch = 250
topology = c(2,3,2)

weightList = initWeightMats(topology)
biasList = initBiasMats(topology, numInputSets)
outputList = forwardProp(input, weightList, biasList)

parameters = SGD(input, weightList, biasList, outputList, trainOutput, learningRate, epoch)

print(forwardProp(matrix(c(0,1), nrow=1, ncol=2, byrow =T), parameters$weights, parameters$biases)$output)
}

iris_mat = as.matrix(as.data.frame(lapply(iris, as.numeric)))
trainInput_mat = apply(iris_mat[c(1:40, 51:90, 101:140), 1:4], 2, function(x) (x - mean(x))/(sd(x)))
validationInput_mat = iris_mat[c(41:50, 91:100, 141:150), 1:4]
trainOutput_vec = iris_mat[c(1:40, 51:90, 101:140),5]
validationOutput_vec = matrix(iris_mat[c(41:50, 91:100, 141:150), 5])

trainOutput_mat = matrix(nrow = length(trainOutput_vec), ncol = 3)

for(i in 1:length(trainOutput_vec)) {
  if(trainOutput_vec[i] == 1) {
    trainOutput_mat[i,] = c(1,0,0) 
  } else if (trainOutput_vec[i] == 2) {
    trainOutput_mat[i,] = c(0,1,0) 
  } else {
    trainOutput_mat[i,] = c(0,0,1) 
  }
}

numTrainingExamples = nrow(trainInput_mat)
numLayers = 3
eluAlpha = .7
learningRate = .25
epoch = 10
topology = c(4,4,3)

weightList = initWeightMats(topology)
biasList = initBiasMats(topology, numTrainingExamples)
outputList = forwardProp(trainInput_mat, weightList, biasList)

parameters = SGD(trainInput_mat, weightList, biasList, outputList, trainOutput_mat, learningRate, epoch)

#print(round(forwardProp(validationOutput_mat, parameters$weights, parameters$biases)))

