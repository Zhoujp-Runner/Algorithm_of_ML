import perceptron
from matplotlib import pyplot as plt
from matplotlib import animation

data = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
per = perceptron.Perceptron(data)
per.sgd(data)





