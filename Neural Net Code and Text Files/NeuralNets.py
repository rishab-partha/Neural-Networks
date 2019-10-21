'''# @author Rishab Parthasarathy
   # @version 9.4.19
   # This file defines a multilayer perceptron. In this state, the
   # neural network defined in this file only supports
   # one hidden layer. However, this neural network
   # can be configured in the future to function with any
   # number of hidden layers. At this point, the neural network
   # takes in inputs and weights, and uses the full
   # connectivity pattern and sigmoids at each
   # activation node to calculate the activations at the node.
   # By passing this data through the neural network,
   # the neural network makes a predicted output.
   #
   # This file also tests the neural network by passing
   # in inputs and outputs and printing out the internals of
   # the perceptron to test its efficacy against known results.
   #'''
import numpy as np
import math
import sys
'''# This class defines a multilayer perceptron.
   # The perceptron is composed of input processing units,
   # hidden layers of activations and an output layer
   # of activations. In between the layers, there are
   # weights that indicate how values propagate through
   # the network. Also, there are nonlinear functions within
   # the nodes that add nonlinearities. The perceptron functions
   # by passing the input values to the input processing units.
   # Then, using the weights and connectivity pattern, the values
   # propagate down to the output units, which define the value that
   # the neural network has now predicted.
   # Currently, the neural network can only evaluate inputs,
   # but in the future, the network will have training
   # and validation functionality.
   #'''
class NeuralNet:
   '''# __init__ defines a multilayer perceptron for execution
      # using the sizes of each layer to define the fully connected
      # connectivity model. Also, it takes in the inputs for execution.
      # Finally, it initializes the activations, outputs, and weights
      # arrays so that they can function without going out of bounds
      # for memory. The constructor also takes in expected outputs
      # and initializes learning factor and change in weights for 
      # training purposes. Currently, the perceptron can only be built 
      # to have three activation layers, so if any other number is indicated,
      # the program quits. Also, if the perceptron is not of a
      # 2-2-1 connectivity pattern, it is currently impossible
      # to model properly.
      #
      # @param self            The neural network being created
      # @param layerSizes      The sizes of each activation layer
      # @param inputs          The inputs that the network will evaluate
      # @param expectedOutputs The expected outputs for the network
      #
      # @precondition  The network has 3 activation layers
      # @postcondition The constructor does not print an error message
      #'''
   def __init__(self, layerSizes, inputs, expectedOutputs, maximumIters, learningFactor, errorThreshold,
                     learningFactorAdjustment, minErrorGap, lambdaMax, lambdaEps, numPrint, randomRange,
                      rWeights, wWeights):
      if len(layerSizes) != 3:
         print("Sorry, but this neural net is not ready for this yet.")
         print("Please try again later.")
         sys.exit()
         # End of error statement

      self.activations = None
      self.inputs = inputs
      self.layerSizes = layerSizes
      self.expectedOutputs = expectedOutputs
      self.learningFactor = learningFactor
      self.maximumIters = maximumIters
      self.errorThreshold = errorThreshold
      self.lambdaMax = lambdaMax
      self.lambdaEps = lambdaEps
      self.learningFactorAdjustment = learningFactorAdjustment
      self.minErrorGap = minErrorGap
      self.numPrint = numPrint
      self.rWeights = rWeights
      self.wWeights = wWeights
      self.randomRange = randomRange
      self.totalError = 2.
      self.outputs = np.zeros(expectedOutputs.shape)
      maximumLayerSize = 0

      for layerSize in layerSizes:
         maximumLayerSize = max(maximumLayerSize, layerSize)
         # End of maximum layer size for loop

      self.activations = np.zeros((len(layerSizes), maximumLayerSize))
      self.weights = np.zeros((len(layerSizes) - 1, maximumLayerSize, maximumLayerSize))
      self.deltaWeights = np.zeros(self.weights.shape)
      self.errors = np.full_like(np.arange(self.inputs.shape[0]), 2.0, dtype = np.double)
      self.setWeights()
      # End of function __init__

   '''# Function setWeights currently functions to hard code the weights to certain values that
      # can be tested within the spreadsheet. In the future, setWeights will read weights in
      # from a text file that will indicate the weight construction of the neural network.
      #
      # @param self   the multilayer perceptron which needs its weights to be hard coded.
      # @precondition The weights array has at least size 2 in all three dimensions
      #'''
   def setWeights(self):
      fileWeights = open(self.rWeights, 'r')
      
      for n in range(len(self.layerSizes) - 1):
         for k in range(self.layerSizes[n]):
            for j in range(self.layerSizes[n + 1]):
               self.weights[n][k][j] = [float(val) for val in fileWeights.readline().split()][0]
      #End of function setWeights

   '''# Function randomizeWeights sets all the weights in the neural network to arbitrary values.
      # This function is not being utilized at this moment, but it will be used in the future for
      # the functionality of choosing a random starting point for training.
      #
      # @param self the multilayer perceptron which needs random weights for training
      #'''
   def randomizeWeights(self):
      self.weights = self.randomRange*np.random.random_sample(self.weights.shape)
      #End of function randomizeWeights

   '''# Function printNet prints out the given state of the multilayer perceptron after processing
      # a certain input. First, the function prints out the input number. Then, it prints out the
      # weights within the neural network. After that, it prints out the activations of all the units
      # within the network itself. Finally, the function prints the given input and the produced output.
      #
      # @param self       the multilayer perceptron which is having its internal readings printed
      # @param inputIndex the index of the input that has just been processed
      #
      # @precondition  inputIndex is the index of the input that was processed immediately before
      #                and inputIndex is an index corresponding to a real input
      # @postcondition printNet prints out the matching internal state to the input at index inputIndex
      #'''
   def printNet(self, inputIndex):
      print("Input #" + str(inputIndex + 1) + ":\n")
      print("Printing weights: ")
      print(self.weights)
      print("Printing activations: ")
      print(self.activations)
      print("Printing input:")
      print(self.inputs[inputIndex])
      print("Printing actual output:")
      print(self.outputs[inputIndex])
      print("Printing expected output:")
      print(self.expectedOutputs[inputIndex])
      #End of function printNet
   
   '''# Function printErrors prints the error for every single test case and also prints
      # out the total error over all testcases.
      #
      # @param self the neural network for which the errors are being printed
      #'''
   def printErrors(self):
      for i in range(len(self.inputs)):
         print(self.errors[i])

      print(self.totalError)
      #End of function printErrors

   '''# Function wrapperFunc outputs the value produced by a sigmoid given an input.
      #
      # @param self     the multilayer perceptron evaluating the sigmoid curve
      # @param inputval the value inputted into the sigmoid curve
      #
      # @return the value generated by the sigmoid for the inputval
      #'''
   def wrapperFunc(self, inputval):
      return 1.0/(1.0+math.exp(-inputval))
      #End of function wrapperFunc

   '''# Function assignInput transfers one of the inputs at a given index to
      # the preliminary activations of the multilayer perceptron. This serves
      # as the input processing unit of the PDP model and allows the neural network
      # to successfully evaluate the input.
      #
      # @param self       the multilayer perceptron where the input is assigned
      # @param inputIndex the index of the input that will be evaluated
      #
      # @precondition     inputIndex is a feasible index in the input array (i.e. the input
      #                   with index inputIndex exists)
      #'''
   def assignInput(self, inputIndex):
      for k in range(self.layerSizes[0]):
         self.activations[0][k] = self.inputs[inputIndex][k]
         #End of for loop and function assignInput

   '''# Function runOneInput evaluates a single input through the neural network.
      # First, the function assigns the input to the initial activations. Then,
      # the function iterates through all the other activations layer by layer to
      # calculate their values by propagating from previous layers. Finally, after
      # the values in the last layer have been calculated, they are transferred to
      # the official outputs array.
      #
      # @param self       the multilayer perceptron where the input is run
      # @param inputIndex the index of the input for the network to evaluate
      #
      # @precondition     inputIndex is a feasible index in the input array (i.e. the input
      #                   with index inputIndex exists)
      # @postcondition    the function produces the expected output for the input as per the
      #                   the spreadsheet
      #'''
   def runOneInput(self, inputIndex):
      self.assignInput(inputIndex)
      #print("Debug: ", end = "")
      for n in range(1, len(self.layerSizes)):
         for j in range(self.layerSizes[n]):
            self.activations[n][j] = 0
            #print("a[" + str(n) + "][" + str(j) + "] = f(", end = "")
            for k in range(self.layerSizes[n - 1]):
               self.activations[n][j] += self.activations[n - 1][k]*self.weights[n - 1][k][j]
               #print("a[" + str(n - 1) + "][" + str(k) + "]w[" + str(n - 1) + "][" + str(k) + "][" + str(j) + "] +", end = "")
               #End of for loop that performs dot product
            #print(")")
            self.activations[n][j] = self.wrapperFunc(self.activations[n][j])
            #End of for loop for wrapper functions and propagating all edges

      for i in range(self.layerSizes[len(self.layerSizes) - 1]):
         self.outputs[inputIndex][i] = self.activations[len(self.layerSizes) - 1][i]
         #End of function runOneInput

   '''# Function runOneBatch evaluates the entire input set through the neural network
      # using function runOneInput. Also, to show the internal structure of the network,
      # function runOneBatch prints out the internal state of the network
      # after the ith input using the function printNet.
      #
      # @param self the multilayer perceptron which is evaluating the input set
      #'''
   def runOneBatch(self):
      for i in range(len(self.inputs)):
         self.runOneInput(i)
         self.printNet(i)
         #End of function runOneBatch

   def runOneBatchWithoutPrint(self):
      for i in range(len(self.inputs)):
         self.runOneInput(i)
         #End of function runOneBatchWithoutPrint
   '''# Calculates the error at a certain input index using traditional square error function.
      #
      # @param self       the multilayer perceptron where the error is calculated
      # @param inputIndex the index of the input to calculate error for
      #'''
   def calculateError(self, inputIndex):
      error = 0
      for i in range(self.outputs.shape[1]):
         whatToSquare = self.outputs[inputIndex][i] - self.expectedOutputs[inputIndex][i]
         error += 0.5*(whatToSquare*whatToSquare)
      
      return error
      #End of function calculateError
   
   '''# Calculates the changes of weights at a certain input index
      # by taking partial derivatives of the error function.
      #
      # @param self the multilayer perceptron where the deltas are calculated
      # @param inputIndex the index of the input for adjustment of weights
      #'''
   def calculateDeltas(self, inputIndex):
      difference = (self.expectedOutputs[inputIndex][0] - self.outputs[inputIndex][0])
      outputDeriv = self.activations[2][0]*(1 - self.activations[2][0])

      for j in range(self.layerSizes[1]):
         partialDeriv = difference*outputDeriv*self.activations[1][j]
         self.deltaWeights[1][j][0] = partialDeriv*self.learningFactor
      
      for k in range(self.layerSizes[0]):
         for j in range(self.layerSizes[1]):
            intermediateDeriv = self.activations[1][j]*(1 - self.activations[1][j])
            partialDeriv = self.activations[0][k]*intermediateDeriv*difference*outputDeriv*self.weights[1][j][0]
            self.deltaWeights[0][k][j] = partialDeriv*self.learningFactor
      #End of function calculateDeltas

   '''# Calculates all the errors by iterating through all
      # the inputs and calculating all the errors.
      #
      # @param self       the multilayer perceptron where the errors are calculated
      #'''
   def calculateAllErrors(self):
      for i in range(len(self.inputs)):
         self.errors[i] = self.calculateError(i)
      #End of function calculateAllErrors

   '''# Calculates the total error by iterating through
      # all the errors and summing and taking square root
      # to calculate total error.
      #
      # @param self       the multilayer perceptron where the errors are calculated
      #'''
   def calculateTotalError(self):
      totalerror = 0

      for i in range(len(self.inputs)):
         totalerror += self.errors[i]*self.errors[i]
      
      self.totalError = math.sqrt(totalerror)
      #End of function calculateTotalError

   '''# Optimizes one error by calculating the deltas for one input and applies the delta
      # and modifies learning factor and moves downhill through steepest descent.
      #
      # @param self the multilayer perceptron which is being optimized
      # @param inputIndex the index of the input for adjustment of weights
      #'''
   def optimizeOneInput(self, inputIndex):
      self.runOneInput(inputIndex)
      self.errors[inputIndex] = self.calculateError(inputIndex)
      self.calculateDeltas(inputIndex)

      #print("Deltas:")
      #print(self.deltaWeights)
      curerror = self.errors[inputIndex]
      #print(curerror)

      for n in range(len(self.layerSizes) - 1):
         for k in range(self.layerSizes[n]):
            for j in range(self.layerSizes[n + 1]):
               self.weights[n][k][j] += self.deltaWeights[n][k][j]
      
      #print("Weights 1.5:")
      #print(self.weights)
      self.runOneInput(inputIndex)
      self.errors[inputIndex] = self.calculateError(inputIndex)

      newerror = self.errors[inputIndex]
      #print(newerror)

      ret = True   #return value that indicates whether the training will continue

      if newerror < curerror:
         self.learningFactor *= self.learningFactorAdjustment

         if self.learningFactor > self.lambdaMax:
            self.learningFactor = self.lambdaMax
      
      elif newerror > curerror:
         self.learningFactor /= self.learningFactorAdjustment

         if self.learningFactor < self.lambdaEps:
            print("Learning Factor became 0.")
            ret = False
         
         for n in range(len(self.layerSizes) - 1):
               for k in range(self.layerSizes[n]):
                  for j in range(self.layerSizes[n + 1]):
                     self.weights[n][k][j] -= self.deltaWeights[n][k][j]

      return ret
      #End of function optimizeOneInput

   '''# Trains one batch by iterating through all inputs and 
      # optimizing the inputs individually.
      #
      # @param self the multilayer perceptron being trained
      #'''
   def trainOneBatch(self):
      #print("Weights 1:")
      #print(self.weights)
      ret = True          #return value that says whether or not to continue training

      for i in range(len(self.inputs)):
         if not self.optimizeOneInput(i):
            ret = False
         #print("Weights 2:")
         #print(self.weights)

      self.runOneBatchWithoutPrint()
      self.calculateAllErrors()
      self.calculateTotalError()

      if (self.totalError < self.errorThreshold):
         print("Network converged at error less than threshold.")
         self.printErrors()
         ret = False

      return ret
      #End of function trainOneBatch
      
   '''# Trains a multilayer perceptron from scratch by using gradient descent.
      #
      # @param self the multilayer perceptron being trained
      #'''
   def train(self):
      willContinue = True
      numIters = 0
      curTotError = self.totalError
      while numIters < self.maximumIters and willContinue:
         willContinue = self.trainOneBatch()
         numIters += 1
         if numIters % self.numPrint == 0:
            print(self.learningFactor)
            print(self.totalError)

            if ((curTotError - self.totalError) < self.minErrorGap):
               print("Network converging too slowly.")
               self.printErrors()
               willContinue = False

            curTotError = self.totalError
      
      if willContinue:
         print("Timed out at " + str(self.maximumIters) + " iterations.")

      else:
         print(numIters)

      fileWeights = open(self.wWeights, 'w+')
      
      for n in range(len(self.layerSizes) - 1):
         for k in range(self.layerSizes[n]):
            for j in range(self.layerSizes[n + 1]):
               fileWeights.write(str(self.weights[n][k][j]) + "\n")

      fileWeights.close()
      #End of function train

'''#
   # Function main serves to configure and test a feed forward multilayer perceptron. It reads
   # in the size of each layer and the input set from two text files, and then uses these to
   # construct a multilayer perceptron of class NeuralNet. Then, it runs the input set through
   # the neural net once to evaluate performance. The layersizes text file is configured in such
   # a way that the first line has the number of layers and the ith line after the first line has
   # the size of the ith layer. The inputs text file is configured such that the first line has the
   # number of inputs and the ith line after the first line has the values in the ith input separated
   # by commas.
   #'''
def main(configFile): 
   fileConfig = open(configFile, 'r')
   
   filehyperParams = open([str(val) for val in fileConfig.readline().split()][0], 'r')
   fileLayerSizes = open([str(val) for val in fileConfig.readline().split()][0], 'r')
   fileInputs = open([str(val) for val in fileConfig.readline().split()][0], 'r') 
   readWeights = [str(val) for val in fileConfig.readline().split()][0] 
   writeWeights = [str(val) for val in fileConfig.readline().split()][0] #Assign file I/O directories

   numLayers = [int(val) for val in fileLayerSizes.readline().split()][0]
   numInputs = [int(val) for val in fileInputs.readline().split()][0]    #Find the number of layers and inputs

   layerSizes = np.zeros((numLayers), np.int32)  #Create the layersizes array

   for i in range(numLayers):
      layerSizes[i] = [int(val) for val in fileLayerSizes.readline().split()][0]
      #End of assigning layersizes

   inputs = np.zeros((numInputs, layerSizes[0]))                            #Create the inputs array
   expectedOutputs = np.zeros((numInputs, layerSizes[len(layerSizes) - 1])) #Create expected outputs array

   for i in range(numInputs):
      ithInput = [int(val) for val in fileInputs.readline().split(", ")]          #Parses ith input
      ithExpectedOutput = [int(val) for val in fileInputs.readline().split(", ")] #Parses ith expected output
      for j in range(layerSizes[0]):
         inputs[i][j] = ithInput[j]
         #End of assigning inputs
      for j in range(layerSizes[len(layerSizes) - 1]):
         expectedOutputs[i][j] = ithExpectedOutput[j]
         #End of assigning expected outputs

   maximumIters = [int(val) for val in filehyperParams.readline().split()][0]
   learningFactor = [float(val) for val in filehyperParams.readline().split()][0]
   errorThreshold = [float(val) for val in filehyperParams.readline().split()][0]
   learningFactorAdjustment = [float(val) for val in filehyperParams.readline().split()][0]
   minErrorGap = [float(val) for val in filehyperParams.readline().split()][0]
   lambdaMax =  [float(val) for val in filehyperParams.readline().split()][0]
   lambdaEps = [float(val) for val in filehyperParams.readline().split()][0]
   numPrint = [int(val) for val in filehyperParams.readline().split()][0]
   randomRange = [float(val) for val in filehyperParams.readline().split()][0]

   nn = NeuralNet(layerSizes, inputs, expectedOutputs, maximumIters, learningFactor, errorThreshold,
                  learningFactorAdjustment, minErrorGap, lambdaMax, lambdaEps, numPrint, randomRange,
                  readWeights, writeWeights)

   nn.randomizeWeights()
   nn.train()
   nn.runOneBatch()
   #End of function main

'''#
   # This statement functions to run the main function by default.
   #'''
if (__name__ == "__main__"):
   main("config.txt")
