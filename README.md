# Plagiarism Notice
Do NOT copy any code from this repository. By using this code, you will be held liable for accusations of academic plagiarism.

# Files
## config.txt
Contains:  
1. Hyperparameter path  
2. Layer Configuration path  
3. Network Input/Output path  
4. Initial Weights path  
5. Final Weights path  
6. Path for a folder to write the outputs
## inputs.txt
Contains number of inputs on first line. Then, every pair of lines has paths of inputs files on the first line and paths of output files on the second line.
## layersizes.txt
Contains the number of layers on the first line. Then, every line after that contains the size of one layer.
## NeuralNets.py
This is the original construction of the single-output trainable neural net for one hidden layer.
### Command Line Prompts
--config_path - the file name of the configuration file
## NeuralNets.pyproj
This gives the ability to configure and use the project on VSCode.
## NeuralNetsMultipleOutput.py
This is the new construction of the multilayer perceptron with one hidden layer but multiple outputs
## traininghyperparams.txt
Contains line by line:  
1. Max #iterations  
2. Initial Learning Factor  
3. Total error to terminate at  
4. Learning Factor Adjustment Rate (what to multiply and divide it by)  
5. Minimum total error decrease over a certain number of iterations  
6. Maximum learning factor  
7. Minimum lambda (essentially epsilon)  
8. Distance between iterations where learning factor and total error are printed and the total error decrease is checked  
9. A path to random range files
10. The range of the inputs so that it can be scaled to between 0 and 1
11. The range of the outputs so that the network can scale outputs between 0 and 1 to the expected output range
## randomrange.txt
Each line contains a comma separated range of values for random values for the nth layer.
## weights0.txt
With the definition of the weights as w[n][k][j], lists the input weights line by line sorted first by n, then k, then j.
## weights1.txt
With the same definition as weights0.txt, weights1.txt represents the weights after training.
