import numpy as np

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
    return parameters

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))  # subtracting the max to avoid numerical instability
    return exp_Z / np.sum(exp_Z, axis=0)

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters[f'W{l}'], A_prev) + parameters[f'b{l}']
        A = relu(Z)
        cache = (A_prev, Z)
        caches.append(cache)
    
    ZL = np.dot(parameters[f'W{L}'], A) + parameters[f'b{L}']
    AL = softmax(ZL)
    cache = (A, ZL)
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, Y):
     
    
    m = Y.shape[1]

     
    cost = -(1/m)*(np.sum(  Y*np.log(AL)+(1-Y)*(np.log(1-AL))  ))
     
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    
    return cost


def backward_propagation(AL, Y, caches, parameters):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Calculate derivative of softmax (assuming AL is the softmax output)
    dZL = AL - Y
    
    # Backpropagate through the layers
    A_prev, ZL = caches[L - 1]
    dW = np.dot(dZL, A_prev.T) / m
    db = np.sum(dZL, axis=1, keepdims=True) / m
    dA_prev = np.dot(parameters[f'W{L}'].T, dZL)
    
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db
    
    for l in reversed(range(L - 1)):
        A_prev, Z = caches[l]
        
        # Use the derivative of ReLU for hidden layers
        dZ = np.array(dA_prev, copy=True)
        dZ[Z <= 0] = 0
        
        # Calculate gradients
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(parameters[f'W{l + 1}'].T, dZ)
        
        grads[f'dW{l + 1}'] = dW
        grads[f'db{l + 1}'] = db
        
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    
    return parameters

def L_layer_model(X, Y, layer_dims, learning_rate=0.01, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layer_dims)
    
    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches,parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
    
    return parameters


# Make predictions
def predict(X, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0)
    return predictions

# Assuming X_test contains your test data


# Assuming predictions is the array of predicted labels and Y_test is the array of actual labels
# You can calculate accuracy as the number of correct predictions divided by the total number of examples

def calculate_accuracy(predictions, Y_test):
    correct_predictions = np.sum(predictions == np.argmax(Y_test, axis=0))
    total_examples = Y_test.shape[1]
    accuracy = correct_predictions / total_examples * 100
    return accuracy

# Assuming predictions and Y_test are numpy arrays
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
# Load the data
df = pd.read_csv('emotions.csv')

# Preprocess the data
s = StandardScaler()
X = s.fit_transform(df.drop(["label"], axis=1))
y = pd.get_dummies(df["label"])  # One-hot encoding using pandas get_dummies

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)

X_train =np.array(X_train).T
X_test=np.array(X_test).T
y_train=np.array(y_train).T
y_test=np.array(y_test).T


# Define the neural network architecture
layer_dims = [X_train.shape[0], 20, 10, y_train.shape[0]]  # Input size, hidden layers, output size

# Train the model
parameters = L_layer_model(X_train, y_train, layer_dims, learning_rate=0.075, num_iterations=1000, print_cost=True)

# Make predictions
predictions_train = predict(X_train, parameters)
predictions_test = predict(X_test, parameters)

# Calculate accuracy

print("Training Accuracy ",calculate_accuracy(predictions_train, y_train))
print("Test Accuracy ",calculate_accuracy(predictions_test, y_test))

