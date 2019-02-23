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

relu = function(x) {
  ifelse(x > 0, x, 0)
}
reluDerivative = function(x) {
  ifelse(x > 0, 1, 0)
}

tanhDerivative = function(x) {
  return (1 - tanh(x)^2)
}

softmax = function(x) {
  sum = sum(exp(x))
  exp(x)/sum
}

forwardProp = function(inputMat, weightMats, biasMats) {
  numSynapses = length(weightMats)
  
  activatedSums = list()
  
  activatedSums[[1]] = relu((inputMat %*% weightMats[[1]]) + biasMats[[1]])
  
  i = 2
  while(i < numSynapses) {
    activatedSums[[i]] = relu((activatedSums[[i-1]] %*% weightMats[[i]]) + biasMats[[i]])
    i = i + 1
  }
  
  output = (activatedSums[[numSynapses-1]] %*% weightMats[[numSynapses]]) 
                                       + biasMats[[numSynapses]]
  
  unactOutput = output
  
  for(i in 1:nrow(output)) {
    output[i,] = softmax(output[i,])
  }
  
  return (list("activatedSums" = activatedSums, "output" = output, "unact" = unactOutput))
}

MSECost = function(targetOutput, netOutput) {
  error = (1/nrow(targetOutput)) * sum( (1/ncol(targetOutput))*(rowSums((targetOutput - netOutput) ^ 2)) )
  return(error)
}

CrossEntropyCost = function(targetOutput, netOutput) {
  error = (1/nrow(targetOutput)) * sum( rowSums(-targetOutput * log(netOutput)))
  return(error)
}

calcAccuracy = function(output, trainOutput) {
  numCorrect = 0
  for(i in 1:length(output)) {
    if (output[i] == trainOutput[i]) {
      numCorrect = numCorrect + 1
    }
  }
  accuracy = numCorrect / length(trainOutput)
  return (accuracy)
}

SGD = function(inputMat, weightList, biasList, outputList, targetOutput, learningRate, epoch, input_test, output_test) {
  origInput_mat = inputMat
  origOutput_mat = targetOutput
  synapseIndex = length(weightList)
  epochNum = 1
  counter = 0
  prevCost = 0
  currCost = 0

  deltaWeightList = list()
  gradWeightList = list()
  gradBiasList = list() #Gradient is same as delta for bias
  
  for(i in 1:synapseIndex) {
    deltaWeightList[[i]] = matrix(nrow=nrow(weightList[[i]]),
                                             ncol=ncol(weightList[[i]]), byrow = T)
    gradWeightList[[i]] = matrix(nrow=nrow(weightList[[i]]),
                                            ncol=ncol(weightList[[i]]), byrow = T)
    gradBiasList[[i]] = matrix(nrow=nrow(biasList[[i]]), 
                                          ncol=ncol(biasList[[i]]), byrow = T)
  }
  
  while(T) {
    for(trainEx in 1:nrow(targetOutput)) {
      outputList = forwardProp(inputMat, weightList, biasList)
      if(anyNA(outputList$output)) {
        print(outputList$activatedSums)
        print(outputList$unact)
        print(outputList$output)
        print("SGD")
        stop()
      }
     
      delta = outputList$output[trainEx,] - targetOutput[trainEx,]
      
      deltaWeightList[[synapseIndex]] = matrix(delta, 
                                               nrow=nrow(weightList[[synapseIndex]]), 
                                               ncol=ncol(weightList[[synapseIndex]]), byrow = T)
      
      for(i in 1:ncol(outputList$activatedSums[[synapseIndex-1]])) {
        gradWeightList[[synapseIndex]][i,] = deltaWeightList[[synapseIndex]][i, ] * outputList$activatedSums[[synapseIndex-1]][trainEx, i]
      }
    
      gradBiasList[[synapseIndex]] = matrix(delta, 
                                            nrow=nrow(biasList[[synapseIndex]]), 
                                            ncol=ncol(biasList[[synapseIndex]]), byrow = T)
      
      while(synapseIndex > 1) {
        synapseIndex = synapseIndex - 1
       
        for(i in 1:nrow(gradWeightList[[synapseIndex]])) {
          for(j in 1:ncol(gradWeightList[[synapseIndex]])) {
            delta = reluDerivative(outputList$activatedSums[[synapseIndex]][trainEx, j]) * 
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
            delta = reluDerivative(outputList$activatedSums[[synapseIndex]][trainEx, j]) * 
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
    
    print(epochNum)
  
    if(epochNum%%10==0 || T)  {
      newOutput = forwardProp(origInput_mat, weightList, biasList)
      #print(round(newOutput$output))
      newBiasList = list()
      for(i in 1:length(biasList)) {
        newBiasList[[i]] = biasList[[i]][1,]
      }
      accuracy = calcAccuracy (round(newOutput$output), origOutput_mat)
      accuracy_test = test(input_test, output_test, weightList, newBiasList)
      plot(epochNum, accuracy, type="b")
      plot(epochNum, accuracy_test, type="b")
      currCost = CrossEntropyCost(origOutput_mat, newOutput$output)
      cat(epochNum, " ", "train: ", as.numeric(accuracy), ", test:  ", as.numeric(accuracy_test), " ", currCost)
      if(abs(currCost - prevCost) < .0001 && F) {
        break
      } else {
        prevCost = currCost
      }
    }
    
    epochNum = epochNum + 1
    randomSwap = sample(1:nrow(inputMat), nrow(inputMat), replace = F)
    
    inputMat = inputMat[randomSwap,]
    targetOutput = targetOutput[randomSwap,]
  }
  
  print("yes")
  newOutput = forwardProp(origInput_mat, weightList, biasList)
  print(round(newOutput$output), digits = 3)
  
  newBiasList = list()
  for(i in 1:length(biasList)) {
    newBiasList[[i]] = biasList[[i]][1,]
  }
  
  return (list("weights" = weightList, "biases" = newBiasList))
  
}

test = function (input_mat, output_mat, weightList, biasList) {
  for(i in 1:length(biasList)) {
    biasList[[i]] = matrix(rep(biasList[[i]], nrow(input_mat)), ncol = length(biasList[[i]]), byrow = T)
  }
  output = forwardProp(input_mat, weightList, biasList)
  return(calcAccuracy(round(output$output), output_mat))
}


index = sample(1:nrow(iris), round(.75*nrow(iris)))
mean_train = apply(iris[index, 1:4], 2, mean)
sd_train = apply(iris[index, 1:4], 2, sd)

data_train = iris[index,]
input_train = scale(data_train[, 1:4], center = mean_train, scale = sd_train)
output_train = matrix(nrow = nrow(input_train), ncol = 3)

for(i in 1:nrow(data_train)) {
  if(as.numeric(data_train[i,5]) == 1) {
    output_train[i,] = c(1,0,0) 
  } else if (as.numeric(data_train[i,5]) == 2) {
    output_train[i,] = c(0,1,0) 
  } else {
    output_train[i,] = c(0,0,1) 
  }
}

data_test = iris[-index,]
input_test = scale(data_test[, 1:4], center = mean_train, scale = sd_train)
output_test = matrix(nrow = nrow(input_test), ncol = 3)
for(i in 1:nrow(data_test)) {
  if(as.numeric(data_test[i,5]) == 1) {
    output_test[i,] = c(1,0,0) 
  } else if (as.numeric(data_test[i,5]) == 2) {
    output_test[i,] = c(0,1,0) 
  } else {
    output_test[i,] = c(0,0,1) 
  }
}
input_test
output_test
data_test
numTrainingExamples = nrow(input_train)
numLayers = 3
#eluAlpha = .7
learningRate = .01
epoch = 100
topology = c(4,72,3)

weightList = initWeightMats(topology)
biasList = initBiasMats(topology, numTrainingExamples)
outputList = forwardProp(input_train, weightList, biasList)

parameters = SGD(input_train, weightList, biasList, outputList, output_train, learningRate, epoch, input_test, output_test)

test(input_test, output_test, parameters$weights, parameters$biases)


