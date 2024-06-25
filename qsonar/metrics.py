import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Class to calculate accuracy, recall, precision, and f-measure for validation and training data
class Metrics:
    def __init__(self):
        self.val_curr = None
        self.val_real = None
        self.qubit_count = None
        self.show_data = []
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None

    def add_data(self, val_curr, val_real, qubit_count):
        self.val_curr = np.array(val_curr)
        self.val_real = np.array(val_real)
        self.qubit_count = qubit_count

        # Calculate TP, TN, FP, FN
        self.TP = np.sum((self.val_curr == 1) & (self.val_real == 1))
        self.TN = np.sum((self.val_curr == 0) & (self.val_real == 0))
        self.FP = np.sum((self.val_curr == 1) & (self.val_real == 0))
        self.FN = np.sum((self.val_curr == 0) & (self.val_real == 1))

        self.show_data.append([self.recall(), self.rmse(), self.mse(), self.accuracy(), self.precision(), self.f_measure(), qubit_count])

    # Recall (Sensitivity)
    def recall(self) -> float:
        if self.TP + self.FN == 0:
            return 0.0
        return self.TP / (self.TP + self.FN)

    # Root Mean Squared Error (RMSE)
    def rmse(self) -> float:
        total_samples = len(self.val_real)
        return np.sqrt(np.sum((self.val_curr - self.val_real) ** 2) / total_samples)

    # Mean Squared Error (MSE)
    def mse(self) -> float:
        total_samples = len(self.val_real)
        return np.sum((self.val_curr - self.val_real) ** 2) / total_samples

    # Accuracy
    def accuracy(self) -> float:
        total_samples = len(self.val_real)
        return (self.TP + self.TN) / total_samples

    # Precision
    def precision(self) -> float:
        if self.TP + self.FP == 0:
            return 0.0
        return self.TP / (self.TP + self.FP)

    # F-measure (F1 score)
    def f_measure(self) -> float:
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # def draw_recall_accuracy(self):
    #
    #     # Extract values and field from show_data
    #     values = [[el[0], el[3]*100] for el in self.show_data] # recall and accuracy
    #     field = [el[-1] for el in self.show_data] # qubit count
    #
    #
    #
    #     print("Values:", values)
    #     print("Field:", field)
    #     plt.scatter(field, [el[0] for el in values], color='red', label='Accuracy')
    #     plt.bar(field, [el[1] for el in values], color='blue', label='Recall')
    #     plt.show()

    def draw_dim_data(self, data, qubits):

        plt.scatter(qubits, [el[0] for el in data], color='red', label='Dimension')
        plt.plot(qubits, [el[0] for el in data], color='red', linestyle='-', linewidth=5)

        plt.bar(qubits, [el[1] for el in data], color='blue', label='data')


        plt.show()

# Test class for Metrics
class MetricsTest:
    def __init__(self):
        pass

    # Calculate and print accuracy, recall, precision, and f-measure for validation data
    def print_metrics(self):
        print("Recall:", round(self.metrics.recall(), 4))
        print("RMSE:", round(self.metrics.rmse(), 4))
        print("MSE:", round(self.metrics.mse(), 4))
        print("Accuracy:", round(self.metrics.accuracy(), 4))
        print("Precision:", round(self.metrics.precision(), 4))
        print("F-measure:", round(self.metrics.f_measure(), 4))
        print("--------------------")

    def runTest(self) -> None:
        self.val_curr = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # predicted validation labels
        self.val_real = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]  # true validation labels
        self.metrics = Metrics()

        self.metrics.add_data(self.val_curr, self.val_real, 1)
        self.print_metrics()

        self.val_curr = [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0]  # predicted labels
        self.val_real = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 2)
        self.print_metrics()

        self.val_curr = [0, 0.5, 0, 0, 0, 1, 1, 0, 1, 0, 0]  # predicted labels
        self.val_real = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 3)
        self.print_metrics()

        self.val_curr = [0, 0.6, 0, 0, 0, 0.6, 0, 0, 1, 1, 1]  # predicted labels
        self.val_real = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 4)
        self.print_metrics()

        self.val_curr = [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0.9]  # predicted labels
        self.val_real = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 5)
        self.print_metrics()

        self.val_curr = [0, 1, 0, 0, 0, 1, 0, 0, 1, 0.3, 0.8]  # predicted labels
        self.val_real = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 6)
        self.print_metrics()

        self.val_curr = [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0.5]  # predicted labels
        self.val_real = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 7)
        self.print_metrics()

        self.val_curr = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0.3]  # predicted labels
        self.val_real = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]  # real labels
        self.metrics.add_data(self.val_curr, self.val_real, 8)
        self.print_metrics()

        # Show graph
        # self.metrics.draw_recall_accuracy()
        self.metrics.draw_dim_data(self.metrics.show_data, [1, 2, 3, 4, 5, 6, 7, 8])

if __name__ == "__main__":
    test = MetricsTest()
    test.runTest()
