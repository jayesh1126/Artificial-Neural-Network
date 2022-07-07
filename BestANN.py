import pandas as pd
from math import exp
import numpy
from random import seed
from random import random
import matplotlib.pyplot as plt


class Neuron:
    # This class will represent represent nodes in our ANN, not the inputs.
    def __init__(self, weights, outputs, delta, momentum):
        self.weights = weights
        self.outputs = outputs
        self.delta = delta
        self.momentum = momentum
    
    def print_neuron(neuron):
        # This is to display all the information about the neuron.
        weight = ', '.join(str(e) for e in neuron.weights)
        moment = ', '.join(str(e) for e in neuron.momentum)
        print("Weights are: " + weight)
        print("Output is: %.5f" %neuron.outputs)
        print("Delta is: %.5f" %neuron.delta)
        print("Momentum values are: " + moment)
      


def initialize_network(input_nodes, hidden_nodes):
    # Here we initiliaze our network, with the number of inputs and the number of nodes in our hidden layer.
    network = []
    # Our network is a list containing 2 lists, one with the neuron of the output layer, and one for the output layer.
    hidden_layer = []
    for i in range(hidden_nodes):
        neuron = Neuron([random()], 0, 0, [0])
        # Creates a neuron with a weights.
        for i in range(input_nodes):
            neuron.weights.append(random())
            # We set the momentum to 0 for the first iteration later.
            neuron.momentum.append(0)
        hidden_layer.append(neuron)
    network.append(hidden_layer)
    output_layer = []
    neuron = Neuron([random()], 0, 0, [0])
    # Creates a neuron with a weights.
    for i in range(hidden_nodes):
        neuron.weights.append(random())
        neuron.momentum.append(0)
    output_layer.append(neuron)
    network.append(output_layer)
    return network

# This is to test and see the network created.
# network = initialize_network(7, 12)
# for layer in network:
#     for neuron in layer:
#        print("neuron has")
#        print(neuron.weights)
#        print(neuron.outputs)
#        print(neuron.delta)


def sigmoid(layerNeuron):
    # Sigmoid function for the ouptut value of a neuron
    return 1.0 / (1.0 + exp(-layerNeuron))

def sigmoid_derivative(output):
    # Calculate the derivative for the delta value
    return output * (1.0 - output)

def sum_weight_inputs(weights, inputs):
    # We calculate the weighted sum for the neuron.
    layerNeuron = weights[-1]
    # The last element of the list of weights is the bias for every neuron.
    for i in range(len(weights)-1):
        # Loops through the neuron's weights.
        layerNeuron += weights[i] * inputs[i]
    return layerNeuron


def forward_propagate(network, row):
    # This is our forward propagation algorithm to give us an output using the inputs.
    inputs = row
    for layer in network:
        # Loops through each layer, hidden and then output.
        new_inputs = []
        for neuron in layer:
            layerNeuron = sum_weight_inputs(neuron.weights, inputs)
            neuron.outputs = sigmoid(layerNeuron)
            # Gets the output of the neuron and adds it to the list of inputs to be used for the next layer.
            new_inputs.append(neuron.outputs)
        inputs = new_inputs
    # At the last iteration, at the output layer we get the output of the whole ANN.
    final_output = inputs[0]
    return final_output


def backward_propagate_error(network, expected):
    # This is our backpropagation algorithm to get the error, and store it in each neurons as a delta value.
    for i in reversed(range(len(network))):
        # We start at the output layer.
        layer = network[i]
        errors = []
        if i == len(network)-1:
             # If we are at the output layer, we just the error of the output and store it.
                neuron = layer[0]
                errors.append(expected - neuron.outputs)
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron.weights[j] * neuron.delta)
                errors.append(error)
        # Set delta values for each neuron in the layer.
        for j in range(len(layer)):
            neuron = layer[j]
            neuron.delta = errors[j] * sigmoid_derivative(neuron.outputs)


def update_weights(network, row, learning_rate):
    # This is our function to update the network's weights using the errors stored in the delta values 
    # of every neuron.
    for i in range(len(network)):
        inputs = row[:-1]
        # Doesn't take into account the last value which is the value we want to predicte (Skelton).
        if i != 0:
            # If we are at the output layer.
            inputs = []
            for neuron in network[i-1]:
                inputs.append(neuron.outputs)
            # The inputs are the outputs of the hidden layer's nodes.
        for neuron in network[i]:
            for j in range(len(inputs)):
                # Updating the weight.
                # We store the old weight to calculate the new momentum value for the neuron.
                old_weight = neuron.weights[j]
                neuron.weights[j] += learning_rate * neuron.delta * inputs[j] + 0.6 * neuron.momentum[j]
                neuron.momentum[j] = neuron.weights[j] - old_weight
            # Now we do the same process for the bias.
            old_bias = neuron.weights[-1]
            neuron.weights[-1] += learning_rate * neuron.delta + 0.6 * neuron.momentum[-1]
            neuron.momentum[-1] = neuron.weights[-1] - old_bias

def simulate_annealing(start_rate, end_rate, epoch, max_epoch):
    # This is our function to decrease the learning rate using annealing.
    new_learning_rate = end_rate + (start_rate - end_rate) * (1-(1/(1+exp((10-((20*epoch)/max_epoch))))))
    return new_learning_rate


def train_network(network, data, learning_rate, epoch, end_learning_rate, Set2):
    # This is our function to train the network we created, by entering the parameters
    #  and the dataset for the training.
    # These lists are to keep track of all the RMSE and MSRE per epoch
    list_RMSE = []
    list_MSRE = []
    # This was to keep track of all the error when predicting after every epochs.
    list_predicted_error = []
    for epochs in range(epoch):
        # We set the new learning rate using annealing.
        new_learning_rate = simulate_annealing(learning_rate, end_learning_rate, epochs+1, epoch)
        list_error = []
        list_relative_error = []
        MSRE = 0
        RMSE = 0
        for row in data:
            # We forward propagate to get an input.
            outputs = forward_propagate(network, row)
            expected = row[-1]
            backward_propagate_error(network, expected)
            # We compare our output to the expected output, and get our error values.
            update_weights(network, row, new_learning_rate)
            # Using the error values we update the weights.
            list_error.append((numpy.square(outputs - expected)))
            list_relative_error.append(numpy.square((outputs-expected)/expected))
        RMSE = numpy.sqrt((sum(list_error)) / (len(list_error)))
        MSRE = (sum(list_relative_error)) / (len(list_relative_error))
        # Here we get the Root Mean Squared Error and Mean Squared Relative Error after each epoch.
        print('Epoch number is: %d, Learning rate is: %.3f, RMSE is : %.3f, MSRE is: %.3f' %
              (epochs, new_learning_rate, RMSE, MSRE))
        # This is to graph all these data
        list_RMSE.append(RMSE)
        list_MSRE.append(MSRE)
        RMSE2 = predict_data_set(network, Set2)
        list_predicted_error.append(RMSE2)
    # predict_data_set(network, Set2)
    # lowest = min(list_predicted_error)
    # print("Lowest predicted error: %.5f at epoch number : %d" %(lowest , list_predicted_error.index(lowest)))
    plt.plot(list_predicted_error, 'g')
    plt.show()


def destandardize(standardized, max_value, min_value):
    # Destandardize our outputs and the expected value to graph it
    destandardized = ((standardized - 0.1)/0.8)*(max_value-min_value)+min_value
    return destandardized

def predict_data_set(network, data):
    # This function allows us to give prediction on a whole data set with our trained network.
    list_error = []
    list_error_destandardized = []
    RMSE = 0
    RMSE_destandardized = 0
    list_out =  []
    list_expected = []
    for row in data:
        outputs = forward_propagate(network, row)
        expected = row[-1]
        destandardized_outputs = destandardize(outputs, 448.1, 3.694)
        destandardized_expected = destandardize(expected, 448.1, 3.694)
        list_out.append(destandardized_outputs)
        list_expected.append(destandardized_expected)
        # This is for testing
        # print("Output is %.3f, Expected is %.3f" %(outputs, expected))
        # print("Output is %.3f, Expected is%.3f" %(destandardized_outputs, destandardized_expected))
        list_error.append((numpy.square(outputs - expected)))
        list_error_destandardized.append(numpy.square(destandardized_outputs - destandardized_expected))
    RMSE = numpy.sqrt((sum(list_error)) / (len(list_error)))
    RMSE_destandardized = numpy.sqrt((sum(list_error_destandardized)) / (len(list_error_destandardized)))
    print('Root mean squared error is: %.3f, Destandardized Root mean squared error is: %.3f' %(RMSE, RMSE_destandardized))
    # plt.scatter(list_out, list_expected)
    # plt.show()
    # plt.plot(list_error_destandardized)
    # plt.show()
    return RMSE_destandardized



# We execute our code using our data sets.
seed(1)
# This is for the random() function to set it between 0 and 1.
TrainingSet = r'C:\Users\jayes\Documents\AI Methods Cw\TrainingSet.csv'
df = pd.read_csv(TrainingSet)
ValidationSet = r'C:\Users\jayes\Documents\AI Methods Cw\Validation.csv'
df1 = pd.read_csv(ValidationSet)
TestSet = r'C:\Users\jayes\Documents\AI Methods Cw\TestSet.csv'
df2 = pd.read_csv(TestSet)
# Getting all the csv data files into different pandas dataframe.
# print(df)
TrainingSet= pd.DataFrame(df).to_numpy()
ValidationSet= pd.DataFrame(df1).to_numpy()
TestSet= pd.DataFrame(df2).to_numpy()
# Transforming the pandas dataframe to numpy arrays.
# numpy.random.shuffle(TrainingSet)
# print(TrainingSet)


input_nodes = len(TrainingSet[0]) - 1
# We initialize our network.
network = initialize_network(input_nodes, 12)
train_network(network, TrainingSet, 0.5, 140, 0.1, ValidationSet)
# We print out our Network.
# for layer in network:
#     print("Layer is:")
#     for neuron in layer:
#         print(neuron)
#         neuron.print_neuron()

# print("Validation set")
# Validation is to tune parameters
# predict_data_set(network, ValidationSet)
# print("test set")
# predict_data_set(network, TestSet)



# This is an extract from the bold driver implementation.
        # for row in data:
        #     count += 1
        #     outputs = forward_propagate(network, row)
        #     expected = row[-1]
        #     backward_propagate_error(network, expected)
        #     # print(outputs)
        #     # print(expected)
        #     bold_driver = 1
        #     old_error = outputs - expected
        #     old_network = network
        #     while bold_driver != 0:
        #         update_weights(network, row, learning_rate)
        #         # Bold driver implementation
        #         bold = forward_propagate(network, row)
        #         new_error = bold - expected
        #         if ((new_error - old_error)/old_error) > 0.04 and count % 100 == 0:  
        #             # If error change is over 4%
        #             learning_rate = learning_rate * 0.7
        #             if learning_rate < 0.01:
        #                 learning_rate = 0.01
        #             # print(learning_rate)
        #             network = old_network
        #             # time.sleep(2)
        #         elif ((new_error - old_error)/old_error) < 0 and ((new_error - old_error)/old_error) > -0.03 and count % 100 == 0:
        #             # If error decreased
        #             learning_rate = learning_rate * 1.05
        #             if learning_rate > 0.5:
        #                 learning_rate = 0.5
        #             bold_driver = 0
        #         else:
        #             # ends the while loop with weights been updated
        #             bold_driver = 0