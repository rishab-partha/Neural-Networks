# Files
## config.txt
Contains:  
1. Hyperparameter path  
2. Layer Configuration path  
3. Network Input/Output path  
4. Initial Weights path  
5. Final Weights path  
## inputs.txt
Contains number of inputs on first line. Then, every pair of lines has comma separated input on the first line and comma separated output on the second line. In the future, instead of comma separated input and outputs, there will be paths to bitmap encoded images.
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
9. Range of random values for random weight generation  
## weights0.txt
With the definition of the weights as w[n][k][j], lists the input weights line by line sorted first by n, then k, then j.
## weights1.txt
With the same definition as weights0.txt, weights1.txt represents the weights after training.