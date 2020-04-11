from owlnet import FeedForward
import numpy as np

nn = FeedForward()
nn.input(2)
nn.layer(2)
nn.initialise()

data = np.array([[1, 0], [0,1], [1,0]])

print(nn.predict(data))
