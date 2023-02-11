Introduction
============

In time series analysis and modeling the underlying properties and behaviours of a time series are very 
important to be understood. These are described mainly by some factors like: the mean, the variance, the 
covariance, the auto and partial auto-correlation coefficients. In this case I was focusing only on the first 
two moments that describes a time series, in particular in the relationship that they have with respect to the 
parameters of the model. 

In general this relationship depends on the structure of the model, on the nature (if determinstic or stochastic), 
on the initial conditions (the initial values) and on the parameters themselves. Once we have a deterministic model 
and we are able to fix the initial conditions, by only changing the parameters we could try to find a way either to: 
come up with a well-formed formula that describes that relationship, or just implement a learning model and experiment with it.