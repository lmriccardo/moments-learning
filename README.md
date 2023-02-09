# First and Second-order moments Learning

FSML is the result of a project for the course *Automatic Verification Of Intelligent Systems* at the University of Rome "La Sapienza" A.A. 2022/2023. The first goal of this project is to create a simple Neural Network that is able to learn how to predict the *mean* and the *variance* from the fixed parameters of a time series model. The second objective is to experiment if it is also possible the inverse. 

A more detailed version that explain the main reasons and the entire procedure that I setted up to complete the project can be found in the PDF report `report.pdf` in the root folder. 

In this README I will just give to you a quick overview of the project and its technical details.

## The Project Idea

In time series analysis and modeling the underlying properties and behaviours of a time series are very important to be understood. These are described mainly by some factors like: the mean, the variance, the covariance, the auto and partial auto-correlation coefficients. In this case I was focusing only on the first two moments that describes a time series, in particular in the relationship that they have with respect to the parameters of the model. 

In general this relationship depends on the structure of the model, on the nature (if determinstic or stochastic), on the initial conditions (the initial values) and on the parameters themselves. Once we have a deterministic model and we are able to fix the initial conditions, by only changing the parameters we could try to find a way either to: come up with a well-formed formula that describes that relationship, or just implement a learning model and experiment with it. 

In this case I decided to implement a simple *Deep Feed-Forward Neural Network* with: 1 input linear layer, 4 hidden linear layer with 50 neurons each, 2 hidden layer with 30 neurons each and 1 output layer. Obviously the number of neurons for the input and the output layer depends on the size of the problem, i.e., the number of parameters and twice the number of outputs of the model. For the training I have used the simple MSE minimization problem using the **Adam** algorithm with some adjustment like: *Cross Fold Validation*, *Adaptive Learning Rate*, *Regularization* and *Data Pre-processing* (standardization). Further details can be found in the PDF report. 