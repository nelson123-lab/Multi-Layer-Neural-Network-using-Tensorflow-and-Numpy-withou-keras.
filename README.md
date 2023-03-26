# Multi-Layer-Neural-Network-using-Tensorflow-and-Numpy-withou-keras.


This is part of Neural Network Course work. The script cannot be posted due to academic integrity. 

The implementaion includes different steps.

The training data had rows as samples and columns as its features.

1) Initialization of weights and bias as a matrix with the bias part of the weight matrix itself ( one extra column ). 
2) Writing functions for different activations.
3) Making a list as weights to keep the weights of all layers with different nodes.
4) Finding the output of each layer and applying different activations after each layer depending on the activation each layer have.

eg: layers = [3, 4, 5, 6]
    Activations = [Relu, Sigmoid, Relu, Linear]
    
    This means that there are 4 layers in the network with 3 nodes in 1st, 4 nodes in 2nd and respectively. The activations of different layers is shown accordingly.
    
5) Applying Gradient Tape to find the gradients and updating the weights.
6) Updating the weights after each epoch and passing the updated weights from one epoch to the next epoch.
7) Testing on the test data with the final updated (weights + bias).
