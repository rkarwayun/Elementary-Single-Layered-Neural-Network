#A Basic Single Layered Neural Network

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        #A single neuron with 3 input connections and 1 output connection.
        #Weights are assigned to a 3x1 matrix with values in the range -1 to 1 and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1


    def sigmoid(self, x):
        return 1/(1+exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    #Training function for neural network
    def train(self, training_inp, training_out, iterations, arr):
        for ite in range(iterations):
            output = self.think(training_inp)   #Pass the training set through Neural Network.

            error = (training_out - output)

            #arr stores the mean error for the three inputs for each iteration.
            arr[ite] = (math.fabs(error[0]) + math.fabs(error[1]) + math.fabs(error[2]))/3

            #Multiply the error by the input again by the gradient of Sigmoid.
            adjustment = dot(training_inp.T, error * self.sigmoid_derivative(output))

            #Adjusting weights.
            self.synaptic_weights += adjustment

    #Function to plot mean error over the total iterations.
    def plott(self, arr):
        plt.plot(arr)
        plt.ylabel('error')
        plt.xlabel('number of iterations')
        plt.show()


    def think(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))



if __name__ == "__main__":

    arr = np.zeros(10000, dtype = float)
    neuralnet = NeuralNetwork()
    print("Random starting weights: ")
    print(neuralnet.synaptic_weights)

    training_inp = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]], dtype=float)
    training_out = np.array([[0,1,1,0]], dtype=float).T

    neuralnet.train(training_inp, training_out, 10000, arr)

    print("New synaptic weights after training: ")
    print(neuralnet.synaptic_weights)

    neuralnet.plott(arr)

    print("Prediction for input [1,0,0]: ")
    print(neuralnet.think(array([1,0,0])))
