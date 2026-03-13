#class to store all functions
import numpy as np

class Functions:
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -60, 60)
        return 1 / (1 + np.exp(-x))