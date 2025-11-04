import random
from variables import *

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, w1=None, b1=None, w2=None, b2=None):
        if w1 and b1 and w2 and b2:
            self.w1 = [[w + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for w in row] for row in w1]
            self.b1 = [b + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for b in b1]
            self.w2 = [[w + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for w in row] for row in w2]
            self.b2 = [b + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for b in b2]
        else:
            self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
            self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
            self.w2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
            self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]

    def activate(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        hidden = []
        for i in range(len(self.w1)):
            val = sum(w * inp for w, inp in zip(self.w1[i], inputs)) + self.b1[i]
            hidden.append(self.activate(val))
        outputs = []
        for i in range(len(self.w2)):
            val = sum(w * h for w, h in zip(self.w2[i], hidden)) + self.b2[i]
            outputs.append(self.activate(val) * 2 - 1)
        return outputs
