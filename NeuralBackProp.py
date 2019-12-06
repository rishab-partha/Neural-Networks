'''# @author Rishab Parthasarathy
   # @version 11.24.19
   # This file defines a multilayer perceptron. In this state, the
   # neural network defined in this file supports
   # any number of hidden layers. At this point, the neural network
   # takes in inputs and weights, and uses the full
   # connectivity pattern and sigmoids at each
   # activation node to calculate the activations at the node.
   # By passing this data through the neural network,
   # the neural network makes a predicted output. In addition,
   # the neural network can be trained to fit a set of inputs to 
   # expected outputs using backpropagation, which uses the partial derivatives
   # of error and intermediate calculations to train all the edges and
   # produce outputs more optimal relative to the expected outputs.
   #
   # This file also tests the neural network by passing
   # in inputs and outputs and printing out the internals of
   # the perceptron to test its efficacy against known results. In addition,
   # the perceptron offers functionality to write to a file in order to extract
   # the information and turn the output values into a bitmap that can be visually
   # interpreted.
   #'''
import numpy as np
import math
import sys
import argparse
import os
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
   # the neural network has now predicted. The network also has training
   # functionality that fits the weights within the network to best associate
   # an input set to an output set, which is performed by modifying the weights
   # through a procedure of gradient based descent.
   #'''
class NeuralNet:
   '''# __init__ defines a multilayer perceptron for execution and training.
      # using the sizes of each layer to define the fully connected
      # connectivity model. Also, it takes in the inputs for execution.
      # Finally, it initializes the activations, outputs, and weights
      # arrays so that they can function without going out of bounds
      # for memory. The constructor also takes in expected outputs
      # and initializes learning factor and change in weights for 
      # training purposes. The constructor finally takes in many hyperparameters,
      # which apart from the learning factor are the minimum error threshold, learning
      # factor adjustment rate, minimum change in error, maximum learning factor, minimum 
      # learning factor, number of iterations between prints, range of randomization for weights,
      # range of adjustment for inputs and outputs, files to write the outputs
      # and files to read and write weights.
      #
      # @param self            The neural network being created
      # @param layerSizes      The sizes of each activation layer
      # @param inputs          The inputs that the network will evaluate
      # @param expectedOutputs The expected outputs for the network
      #
      #'''
   def __init__(self, layerSizes, inputs, expectedOutputs, maximumIters, learningFactor, errorThreshold,
                     learningFactorAdjustment, minErrorGap, lambdaMax, lambdaEps, numPrint, randomRange, inputRange, outputRange,
                      rWeights, wWeights, wOutputs):

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
      self.inputRange = inputRange
      self.outputRange = outputRange
      self.totalError = 500.
      self.outputs = np.zeros(expectedOutputs.shape)
      self.wOutputs = wOutputs

      self.inputs /= inputRange
      self.expectedOutputs /= outputRange

      maximumLayerSize = 0

      for layerSize in layerSizes:
         maximumLayerSize = max(maximumLayerSize, layerSize)
         # End of maximum layer size for loop

      self.activations = np.zeros((len(layerSizes), maximumLayerSize))
      self.weights = np.zeros((len(layerSizes) - 1, maximumLayerSize, maximumLayerSize))
      self.deltaWeights = np.zeros(self.weights.shape)
      self.errors = np.full_like(np.arange(self.inputs.shape[0]), 2.0, dtype = np.double)
      self.thetas = np.zeros(self.activations.shape)
      self.psis = np.zeros(maximumLayerSize)
      self.omegas = np.zeros((2, maximumLayerSize))
   
      # End of function __init__

   '''# Function setWeights reads weights in from a text file that indicate the weight construction 
      # of the neural network.
      #
      # @param self   the multilayer perceptron which needs its weights to be read in.
      # @precondition The weights array has at least size 2 in all three dimensions
      #'''
   def setWeights(self):
      fileWeights = open(self.rWeights, 'r')
      
      for n in range(len(self.layerSizes) - 1):
         for k in range(self.layerSizes[n]):
            for j in range(self.layerSizes[n + 1]):
               self.weights[n][k][j] = [float(val) for val in fileWeights.readline().split()][0]
      #End of function setWeights

   '''# Function randomizeWeights sets all the weights in the neural network to arbitrary values
      # in a certain range.
      #
      # @param self the multilayer perceptron which needs random weights for training
      #'''
   def randomizeWeights(self):
      for n in range(len(self.layerSizes) - 1):
         randRange = self.randomRange[n][1] - self.randomRange[n][0]
         self.weights[n] = randRange*np.random.random_sample(self.weights[n].shape) + self.randomRange[n][0]
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
   
   def printOutputs(self, inputIndex):
      if not os.path.exists(self.wOutputs):
         os.mkdir(self.wOutputs)
      outputFile = open(os.path.join(self.wOutputs, "output" + str(inputIndex) + ".txt"), 'w')
      for i in range(self.layerSizes[len(self.layerSizes) - 1]):
         outputFile.write(str(int(self.activations[len(self.layerSizes) - 1][i] * self.outputRange)) + '\n')

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
      # the official outputs array. In addition, the function runOneInput saves the 
      # thetas, or dot products for each node. This is because these thetas become
      # integral in the backpropagation algorithm.
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
      #print("Debug: ", end = "") #DEBUG
      for n in range(1, len(self.layerSizes)):
         for j in range(self.layerSizes[n]):
            self.thetas[n][j] = 0.0

            #print("a[" + str(n) + "][" + str(j) + "] = f(", end = "") #DEBUG

            for k in range(self.layerSizes[n - 1]):
               self.thetas[n][j] += self.activations[n - 1][k]*self.weights[n - 1][k][j]

               #print("a[" + str(n - 1) + "][" + str(k) + "]w[" + str(n - 1) +  #DEBUG
               #  "][" + str(k) + "][" + str(j) + "] +", end = "")

               #End of for loop that performs dot product
            
            #print(")")

            self.activations[n][j] = self.wrapperFunc(self.thetas[n][j])
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
         self.printOutputs(i)
         #End of function runOneBatch

   '''# Function runOneBatchWithoutPrint evaluates the entire input set through the neural network
      # using function runOneInput.
      #
      # @param self the multilayer perceptron which is evaluating the input set
      #'''
   def runOneBatchWithoutPrint(self):
      for i in range(len(self.inputs)):
         self.runOneInput(i)
         #End of function runOneBatchWithoutPrint

   '''# Calculates the error at a certain input index using traditional mean squared error (MSE) function. 
      # This means the difference between expected and produced is squared for each output and divided by
      # two before all being added together for the final error value.
      #
      # @param self       the multilayer perceptron where the error is calculated
      # @param inputIndex the index of the input to calculate error for
      #'''
   def calculateError(self, inputIndex):
      error = 0.0
      for i in range(self.outputs.shape[1]):
         whatToSquare = self.outputs[inputIndex][i] - self.expectedOutputs[inputIndex][i]
         error += 0.5*(whatToSquare*whatToSquare)
      
      return error
      #End of function calculateError
   
   '''# Calculates the derivative of the activation function based on the value of the
      # input itself
      #
      # @param self the multilayer perceptron where the derivative is being calculated
      # @param inputval the value of the input whose derivative is being calculated
      #'''
   def calculateDerivs(self, inputVal):
      x = self.wrapperFunc(inputVal)
      return x*(1.0 - x)

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
      # all the errors, finding the average of the squares of the individual errors, 
      # and taking square root to calculate total error.
      #
      # @param self       the multilayer perceptron where the errors are calculated
      #'''
   def calculateTotalError(self):
      totalerror = 0.0

      for i in range(len(self.inputs)):
         totalerror += self.errors[i]*self.errors[i] / float(len(self.inputs))
            
      self.totalError = math.sqrt(totalerror)
      #End of function calculateTotalError

   '''# Optimizes one error by calculating the deltas for one input and applies the delta
      # and modifies learning factor and moves downhill through steepest descent. These deltas are calculated
      # dynamically using the backpropagation algorithm, and the weights themselves are dynamically updated. If the
      # steepest descent goes downhill, learning factor is augmented, but if the steepest descent stagnates,
      # learning factor is not augmented. Also, if steepest descent goes uphill, learning factor is decreased
      # and the weight changes are reversed. Finally, if learning factor becomes zero, training terminates and
      # the cause is printed.
      #
      # @param self the multilayer perceptron which is being optimized
      # @param inputIndex the index of the input for adjustment of weights
      #'''
   def optimizeOneInput(self, inputIndex):
      self.runOneInput(inputIndex)
      self.errors[inputIndex] = self.calculateError(inputIndex)

      for i in range(self.layerSizes[len(self.layerSizes) - 1]):
         self.omegas[0][i] = (self.expectedOutputs[inputIndex][i] - self.outputs[inputIndex][i])
      
      for n in range(len(self.layerSizes) - 2, -1, -1):

         for j in range(self.layerSizes[n + 1]):
            self.psis[j] = self.omegas[0][j]*self.calculateDerivs(self.thetas[n + 1][j])

            for k in range(self.layerSizes[n]):
               self.omegas[1][k] += self.psis[j]*self.weights[n][k][j]
               self.deltaWeights[n][k][j] = self.learningFactor*self.activations[n][k]*self.psis[j]
               self.weights[n][k][j] += self.deltaWeights[n][k][j]
         
         self.omegas[0] = self.omegas[1].copy()
         self.omegas[1] = np.zeros(len(self.omegas[0]))


      #print("Deltas:")  #DEBUG
      #print(self.deltaWeights)

      curerror = self.errors[inputIndex]

      #print(curerror) #DEBUG

      #print("Weights 1.5:") #DEBUG
      #print(self.weights)

      self.runOneInput(inputIndex)
      self.errors[inputIndex] = self.calculateError(inputIndex)
      newerror = self.errors[inputIndex]

      #print(newerror) #DEBUG

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
      # optimizing the inputs individually by using gradient descent with backpropagation.
      # Also, this function runs one batch at the end to find the new total error, and
      # if the total error has converged under a respectable threshold, it terminates
      # training and prints out the reasoning.
      #
      # @param self the multilayer perceptron being trained
      #'''
   def trainOneBatch(self):
      #print("Weights 1:") #DEBUG
      #print(self.weights)

      ret = True          #return value that says whether or not to continue training

      for i in range(len(self.inputs)):
         if not self.optimizeOneInput(i):
            ret = False

         #DEBUG: print("Weights 2:") #DEBUG
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
      
   '''# Trains a multilayer perceptron from scratch by using gradient descent with backpropagation. Every
      # certain amount of iterations, the training prints out learning factor and total error.
      # At the same time, the training checks if the error has decreased enough, and if not,
      # it terminates the training. At the end of the training iterations, it prints out the 
      # number of iterations and writes the weights to a file. If the neural net terminates
      # based on conditions defined in this function, it prints the reason.
      #
      # @param self the multilayer perceptron being trained
      #'''
   def train(self):
      willContinue = True
      numIters = 0
      curTotError = self.totalError
      print(self.learningFactor)
      while numIters < self.maximumIters and willContinue:
         willContinue = self.trainOneBatch()
         numIters += 1

         if numIters % self.numPrint == 0:
            print(self.learningFactor)
            print(self.totalError)

            if ((curTotError - self.totalError) < self.minErrorGap):
               print("Network converging too slowly.")
               print(self.minErrorGap)
               self.printErrors()
               willContinue = False

            curTotError = self.totalError
      
      if willContinue:
         print("Timed out at " + str(self.maximumIters) + " iterations.")

      else:
         print(numIters)
         print("Error Threshold = " + str(self.errorThreshold))

      fileWeights = open(self.wWeights, 'w+')
      
      for n in range(len(self.layerSizes) - 1):
         for k in range(self.layerSizes[n]):
            for j in range(self.layerSizes[n + 1]):
               fileWeights.write(str(self.weights[n][k][j]) + "\n")

      fileWeights.close()
      #End of function train

'''#
   # Function main serves to configure and test a feed forward multilayer perceptron. It reads
   # all the file paths from a configuration file and then reads in the layer sizes, 
   # inputs and expected output. Afterwards, those are used to construct a multilayer perceptron of class NeuralNet. 
   # Then, it randomizes the weights in the neural net and trains on the input set, and after training terminates,
   # the inputs are run through the neural net once to evaluate performance. 
   # The details of all the text files are in the README.
   #
   # @param configFile the configuration file that indicates the files for input and output
   #'''
def main(configFile): 
   fileConfig = open(configFile, 'r')
   
   filehyperParams = open([str(val) for val in fileConfig.readline().split()][0], 'r')
   fileLayerSizes = open([str(val) for val in fileConfig.readline().split()][0], 'r')
   fileInputs = open([str(val) for val in fileConfig.readline().split()][0], 'r') 
   readWeights = [str(val) for val in fileConfig.readline().split()][0] 
   writeWeights = [str(val) for val in fileConfig.readline().split()][0] 
   writeOutputs = [str(val) for val in fileConfig.readline().split()][0] #Assign file I/O directories

   numLayers = [int(val) for val in fileLayerSizes.readline().split()][0]
   numInputs = [int(val) for val in fileInputs.readline().split()][0]    #Find the number of layers and inputs

   inputFile = None
   outputFile = None

   layerSizes = np.zeros((numLayers), np.int32)  #Create the layersizes array
   randomRange = np.zeros((numLayers - 1, 2))
   for i in range(numLayers):
      layerSizes[i] = [int(val) for val in fileLayerSizes.readline().split()][0]
      #End of assigning layersizes

   maximumIters = [int(val) for val in filehyperParams.readline().split()][0]
   learningFactor = [float(val) for val in filehyperParams.readline().split()][0]
   errorThreshold = [float(val) for val in filehyperParams.readline().split()][0]
   learningFactorAdjustment = [float(val) for val in filehyperParams.readline().split()][0]
   minErrorGap = [float(val) for val in filehyperParams.readline().split()][0]
   lambdaMax =  [float(val) for val in filehyperParams.readline().split()][0]
   lambdaEps = [float(val) for val in filehyperParams.readline().split()][0]
   numPrint = [int(val) for val in filehyperParams.readline().split()][0]
   randomRangeFile = open([str(val) for val in filehyperParams.readline().split()][0], 'r')

   for i in range(numLayers - 1):
      randomRange[i] = [float(val) for val in randomRangeFile.readline().split(', ')]

   inputRange =  [float(val) for val in filehyperParams.readline().split()][0]       
   outputRange = [float(val) for val in filehyperParams.readline().split()][0] #End of reading hyperparameters

   inputs = np.zeros((numInputs, layerSizes[0]))                            #Create the inputs array
   expectedOutputs = np.zeros((numInputs, layerSizes[len(layerSizes) - 1])) #Create expected outputs array

   for i in range(numInputs):
      ithInputFile = open([str(val) for val in fileInputs.readline().split()][0])        #Parses ith input

      for j in range(layerSizes[0]):
         inputs[i][j] = [float(val) for val in ithInputFile.readline().split()][0]
         #End of assigning inputs
         
      ithOutputFile = open([str(val) for val in fileInputs.readline().split()][0])
      for j in range(layerSizes[len(layerSizes) - 1]):
         expectedOutputs[i][j] = [float(val) for val in ithOutputFile.readline().split()][0]
         #End of assigning expected outputs


   nn = NeuralNet(layerSizes, inputs, expectedOutputs, maximumIters, learningFactor, errorThreshold,
                  learningFactorAdjustment, minErrorGap, lambdaMax, lambdaEps, numPrint, randomRange, inputRange, outputRange,
                  readWeights, writeWeights, writeOutputs)

   nn.randomizeWeights()
   nn.train()
   nn.runOneBatch()
   #End of function main

'''#
   # This statement functions to run the main function by default. This statement also encodes
   # a command line argument to define the configuration file, with default value config.txt.
   #'''
if (__name__ == "__main__"):
   commandLineArgs = argparse.ArgumentParser()
   commandLineArgs.add_argument("--config_path", type = str, default = "config.txt")
   arguments = commandLineArgs.parse_args() 
   main(arguments.config_path)
