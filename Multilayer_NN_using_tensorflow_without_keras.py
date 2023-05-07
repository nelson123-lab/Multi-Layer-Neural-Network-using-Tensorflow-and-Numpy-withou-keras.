import numpy as np
import tensorflow as tf

# Function to initialize weights as tensors.
def Weights_initializer(layers, X_train, seed):
  # layers : The no of layers in the Neural Network
  # X_train : The X_train of the data
  # seed : The seed value of the randamization

  # Initializing the weights as an empty list to append all the initial weights if weights are nto given.
  Weights = []

  # Iterating over the layers to find the weights of each layer.
  for idx, ele in enumerate(layers):

    # Taking the seed value of the randamization.
    np.random.seed(seed)

    # The intial weights are taken according to the shape of the inputs.
    # Weights in the first layer is of the dimension of (no of features, no of nodes).
    if idx == 0: Weights.append(tf.Variable(np.random.randn(X_train.shape[1] +1, ele), dtype = tf.float32))

    # The weights in the other layers are taken depending on the previous layers.
    # Weights in other layers are of the dimension of (no of nodes of previous layer, no of nodes of current layer).
    else: Weights.append(tf.Variable(np.random.randn(layers[idx-1]+1, ele), dtype = tf.float32))

    # Returing the weights list with weight tensors of each layer within it.
  return Weights

# Code given by TA Jason as helper files for splitting the data according to the validation split.
def split_data(X_train, Y_train, split_range=[0.2, 0.7]):
    # Initializing the start and end
    start, end = int(split_range[0] * X_train.shape[0]), int(split_range[1] * X_train.shape[0])

    # returning X_train, y_train, X_val, y_val
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate((Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]

# Code given by TA Jason as helper files for converting the training and testing data into different batches according to batch size.
def generate_batches(X, y, batch_size=32):
    # Iterating over the X for splitting the data.
    for i in range(0, X.shape[0], batch_size):
        # Generating each batcha at a time.
        yield X[i:i+batch_size], y[i:i+batch_size]

    # Handling the left over data when there is extra batches that was already taken in the first part.
    """
    The below part of the generate_batches function can be removed inorder to handle the case where the only the batchs are needed.
    The part of the data that was already taken by the first part of the code is taken again in the second part.
    """

    # Handling the left over data again.
    if X.shape[0] % batch_size != 0: yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]
        

# Function to return the shape normalized dimesions by adding a row of ones to them.
def shape_normalizer(X):
  # X : The data to which ones should be added inorder to match which the dimesions.
  # A row of ones is concatenated to the data.
  return tf.concat([tf.ones([1, X.shape[1]], dtype = tf.float32), X], axis = 0)


# Function to apply activation on each layers output.
def applying_activation(Z, Activation):
    # Z : The data to which activations should be added
    # Activation : The activations can sigmoid, relu or linear.

    # Returing the output after applying the activations.
    return tf.nn.sigmoid(Z) if Activation.lower() == 'sigmoid' else (tf.nn.relu(Z) if Activation.lower() == 'relu' else Z)

def layer_output_finder(layer_weights, X_train_sample, layer_activations):
  # layer_weights : Weights of each layers, it can be updated weights or the final weights of the fully trained neural network.
  # X_train_sample : The training samples one at a time
  # layers : The no of layers in the Neural network when helps in finding the outputs in each layer.

  # Outputs of each layer in neural network is stored in this list.
  Outputs_layers = [] 

  # Iterating over the layer weights.
  for idx, ele in enumerate(layer_weights):

    # Output of first layer depends on the inputs
    if len(Outputs_layers) == 0: 
       
       # Appylying matrix multiplication with the transpose of the weights and the shape normalized transpose of the X_training sample.
       Output = tf.matmul(tf.transpose(ele), shape_normalizer(tf.transpose(X_train_sample)))

       # Appending the output of the first layer to use it as the input in the next layer.
       Outputs_layers.append(applying_activation(Output, layer_activations[idx]))

    # Outputs of next layer depends on the previous layers
    else: 
       # Appylying matrix multiplication with the transpose of the weights and the shape normalized outputs from the previous layer.
       Output = tf.matmul(tf.transpose(ele), shape_normalizer(Outputs_layers[-1]))

       # Appending the output of the current layer to use it as the input in the next layer.
       Outputs_layers.append(applying_activation(Output, layer_activations[idx]))

    # Returing the tranpose of the last layers output to match with the dimensions of the target.
  return tf.transpose(Outputs_layers[-1])

# The function to find the loss values.
def loss_function(Actual, Predicted, loss):
    # loss: the loss can svm, mse, or cross_entropy.
    
    # Returing the svm loss if loss == 'svm'
    if loss.lower() == 'svm': return tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, (1.0 - (Actual * Predicted))), axis=1))

    # Returning the MSE when loss == 'mse'
    elif loss.lower() == 'mse': return tf.reduce_mean(tf.square(Actual - Predicted))

    # Returning the Cross Entropy loss when the loss == 'cross_entropy'
    elif loss.lower() == 'cross_entropy': return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Actual, logits = Predicted))
    
    # Else pass
    else: pass

# Function for fitting the data which provides the weights+bias after training.
def fit(X, y, weights, layer_activations, loss, alpha):
    # Using tf.GradientTape to find the gradients for each layers.
    with tf.GradientTape() as tape:

        # Keeping track of the weights and bias
        tape.watch(weights) 

        # Finding the output of for the X
        predicted = layer_output_finder(weights, X, layer_activations)
        
        # Finding the error according to the preferred loss method by comparing the actual and predicted.
        error = loss_function(y,predicted, loss)

    # Storing the grdients matrix for each layer in a list of gradients.
    gradients = tape.gradient(error, weights)
    
    # Iterating over each weights of each layer.
    for i in range(len(weights)):

        # Updating the new weights for each layer.
        weights[i].assign_sub(alpha * gradients[i])

    # Returing the updated weights.
    return weights

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):
    # Splitting the data into X_train, y_train, X_val, y_val
    X_train, y_train, X_val, y_val = split_data(X_train, Y_train, validation_split)

    # Converting the training data into tensors.
    X_train, y_train = tf.convert_to_tensor(X_train, dtype = tf.float32), tf.convert_to_tensor(y_train, dtype = tf.float32)

    # Converting the validation data into tensors.
    X_val, y_val = tf.convert_to_tensor(X_val, dtype = tf.float32), tf.convert_to_tensor(y_val, dtype = tf.float32)

    # The error list to which the final errors are added.
    err = []

    # Assigning the weights if weights are already given else random intialization.
    W = weights if weights is not None else Weights_initializer(layers, X_train, seed)

    # Initializing the weights list to store the updated weights per epooch.
    w_list = [W]

    # Iterating over the epochs.
    for i in range(epochs):

        # Iterating over the batches.
        for X_batch, y_batch in generate_batches(X_train, y_train, batch_size):

            # Traing to find the updated weights.
            U_w = fit(X_batch, y_batch, w_list[-1], activations, loss, alpha)

            # The error list to which the errors in each batch is stored.
            Error_list = []

            # Finding the validation output with the updated weights.
            y_pred = layer_output_finder(U_w, X_val, activations)

            # Appending the error values to the error list.
            Error_list.append(loss_function(y_val, y_pred, loss))
        
        # Appending the errors per epoch to the err.
        err.append(tf.reduce_mean(Error_list))
    
    # Finding the final output of X_val using the trained weights and storing the final weights in the W.
    Out, W = layer_output_finder(w_list[-1], X_val, activations), w_list[-1]

    # Returing the Trained weights, errors per epoch, Final output.
    return [W, err, Out]

