import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''


def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer
    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer
    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label


def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''

    s = 1.0 / (1.0 + np.exp(-z))

    return s


def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Your code here

    # For viewing entire matrix contents
    np.set_printoptions(threshold=np.inf)

    # print("original data : ")
    # print(train_data)
    # print("transformed data : ")

    # get the number of image samples (rows of input)
    num_image_samples = train_label.shape[0]

    # create a zeroed matrix for one of k encoding (to be used later with train_label vector)
    # From piazza example (# rows, # cols)
    one_of_k_matrix = np.zeros((num_image_samples, n_class))

    # TODO: Forward propagation

    # Add bias of 1 for dot product with weight vector (1*W[D])
    # train_data = np.column_stack((train_data, np.ones(num_image_samples)))
    #print(np.ones(num_image_samples).shape)
    bias_column = np.ones(num_image_samples)
    bias_column = bias_column[:,np.newaxis]

    # Append additional bias column to the data matrix
    train_data = np.append(train_data, bias_column, 1)

    # hidden layer matrix Z
    netZ = np.dot(train_data, np.transpose(W1))
    Z = sigmoid(netZ)

    # Add bias of 1 for dot product with weight vector (1*W[M])
    num_hidden = Z.shape[0]
    bias_column = np.ones(num_hidden)
    bias_column = bias_column[:,np.newaxis]

    # Append additional bias column to the hidden matrix
    Z = np.append(Z, bias_column, 1)

    # net output matrix O
    netO = np.dot(Z, np.transpose(W2))
    O = sigmoid(netO)

    # TODO: Convert training data into 1 of K matrix encoding

    # for each row specified by the index label,
    # place a 1 in for the corresponding row value from train_label
    # get label value of every sample (train label)
    rows = np.arange(num_image_samples)
    # Via piazza example
    one_of_k_matrix[rows, train_label] = 1

    # 1 of k matrix, now with expected (training) values marked
    train_k_matrix = one_of_k_matrix

    # Compute the error for the output matrix
    # given by equation 9
    O_error = (O - train_k_matrix)

    # TODO: Backpropagation

    # compute the gradient for weight matrix 2
    # Given by equation 8 :
    W2_gradient = np.dot(np.transpose(O_error), Z)

    # given by equation 12 :
    left = (1 - Z) * Z
    right = np.dot(O_error, W2)
    W1_gradient = np.dot(np.transpose(left * right), train_data)

    # remove junk gradient row (bias result)
    W1_removed = np.delete(W1_gradient, n_hidden, 0)

    # As suggested, flatten the resulting gradient's:
    objective_gradient = np.concatenate((W1_removed.flatten(), W2_gradient.flatten()), 0)

    # divide by the number of image samples
    obj_grad = objective_gradient / num_image_samples

    # TODO: Regularization

    # negative log-likelihood error function (equation 5) :
    left = train_k_matrix * np.log(O)
    right = (1 - train_k_matrix) * np.log(1 - O)
    objective_value = np.sum(left + right)
    objective_value = objective_value * (-1 / num_image_samples)

    # Regularization equation to help reduce overfitting (given by equation 15) :
    left = (lambdaval / (2 * num_image_samples))
    right = (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    regularized_val = left * right
    # sum the results for final error function value
    obj_val = objective_value + regularized_val

    print("Objective value is : ")
    print(obj_val)

    return obj_val, obj_grad


def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.
    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image
    % Output:
    % label: a column vector of predicted labels
    '''

    # Your code here

    # TODO: Forward pass

    # get the number of image samples (rows of input)
    num_image_samples = data.shape[0]

    # Add bias of 1 for dot product with weight vector (1*W[D])
    # train_data = np.column_stack((train_data, np.ones(num_image_samples)))
    bias_column = np.ones(num_image_samples)
    bias_column = bias_column[:,np.newaxis]

    # Append additional bias column to the data matrix
    data = np.append(data, bias_column, 1)

    # hidden layer matrix Z
    netZ = np.dot(data, np.transpose(W1))
    Z = sigmoid(netZ)

    # Add bias of 1 for dot product with weight vector (1*W[M])
    num_hidden = Z.shape[0]
    bias_column = np.ones(num_hidden)
    bias_column = bias_column[:,np.newaxis]

    # Append additional bias column to the hidden matrix
    Z = np.append(Z, bias_column, 1)

    # net output matrix O
    netO = np.dot(Z, np.transpose(W2))
    O = sigmoid(netO)

    # Return the index (vector) with highest probability of prediction
    label = np.argmax(O, axis=1)
    print(label)

    return label
