import numpy as np

class FeedForward():

    accepted_activations = ['sigmoid']

    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
        self.activations = []
        self.input_dims = 0

    def input(self, n: int) -> None:
        self.input_dims = n

    def layer(self, n: int, activation:str='sigmoid') -> None:
        assert(isinstance(n, int))
        assert(activation in self.accepted_activations)
        assert(self.has_input())
        self.layers.append({'dim':n, 'activation':activation})

    def has_input(self) -> None:
        if self.input_dims > 0:
            return True
        else:
            return False

    def initialise(self) -> None:
        for i, layer in enumerate(self.layers):
            W = np.random.rand(self.layers[i-1]['dim'], layer['dim']) / 10
            self.weights.append(W)

            b = np.zeros((layer['dim'], 1))
            self.biases.append(b)

            activation = getattr(self, 'activationfn_' + layer['activation'])
            self.activations.append(activation)

    def predict(self, X: np.array) -> (float, np.array):
        if len(X.shape) == 1:
            X = X.reshape((1, len(X)))
        n, p = X.shape
        assert(p == self.input_dims)

        H = X.T
        print("Input vector", H)
        for i, layer in enumerate(self.layers):
            activation = self.activations[i]
            W = self.weights[i]

            b = self.biases[i]

            H = activation( W @ H + b )

        return H

    @staticmethod
    def activationfn_sigmoid(x: np.array) -> np.array:
         return 1 / (1 + np.exp(x))
