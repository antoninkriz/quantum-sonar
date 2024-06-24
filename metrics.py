import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit_ibm_runtime
from qiskit_aer import AerSimulator

class Metrics:
    def __init__(self, train_loss, val_loss, train_acc, val_acc):
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_acc = train_acc
        self.val_acc = val_acc

    def Recall(self, y_true, y_pred):
        pass


# Create an instance of Metrics
