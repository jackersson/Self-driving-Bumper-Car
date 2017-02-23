import random
import numpy as np


def sigmoid(z): #activation function   
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z): #activation function derivative    
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes, output_function=sigmoid, output_derivative=sigmoid_prime):
        """
        sizes : neurons count in each layer, example - [2, 3, 1]
        biases and weights are randomly initialized
        output_function: cost function
        output_derivative: cost function derivative      
        """

        # coefficients for L1, L2 regularization
        self.l1 = 0.0005 #["0", "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
        self.l2 = 0.0001 #["0", "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        assert output_function is not None, "You should either provide output function or leave it default!"
        self.output_function = output_function
        assert output_derivative is not None, "You should either provide derivative of the output function or leave it default!"
        self.output_derivative = output_derivative

    def feedforward(self, a):
        """
        forward_pass
        """
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)

        output = np.dot(self.weights[-1], a) + self.biases[-1]
        output = self.output_function(output)

        return output

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):     

        if test_data is not None: n_test = len(test_data)
        n = len(training_data)
        success_tests = 0
        for j in range(epochs):
            #random.shuffle(training_data) #randomize the data order
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                success_tests = self.evaluate(test_data)
                print("Эпоха {0}: {1} / {2}".format(
                    j, success_tests, n_test))
            #else:
                #print("Epoch {0} complete".format(j))
        if test_data is not None:
            return success_tests / n_test

    def update_mini_batch(self, mini_batch, eta):       
        #TODO consider ADAM, Momentum
        #TODO consider learning rate decay
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta / len(mini_batch)
        #added regularization
        self.weights = [w - eps * nw - self.l1 * np.sign(w) - self.l2 * w for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - eps * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):        

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = x
        activations = [x]  # all activations layer by layer
        zs = []  # z vectors -> layer by layers (sum)

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        output = self.output_function(z)
        activations.append(output)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.output_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
      
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def evaluate(self, test_data):        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):       
        return output_activations - y
