import nnFunctions
import numpy as np

# Paste your sigmoid function here


# Paste your nnObjFunction here


n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace(-5,5, num=26)
args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)
objval,objgrad = nnFunctions.nnObjFunction(params, *args)
print("Objective value:")
print(objval)
print("Gradient values: ")
print(objgrad)