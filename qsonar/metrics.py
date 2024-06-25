import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Class to calculate accuracy, recall, precision, and f-measure for validation and training data
class Metrics:
    def __init__(self, name: str):
        '''
        Metrics class for calculating accuracy, recall, precision, and f-measure for validation and training data.
        Pipeline:
        1. Create Metrics object.
        2. Add data of current and real values for 60 qubits (dimensions).
        3. Print/show metrics for each dimension.
        '''

        self.name = name
        self.val_curr = None
        self.val_real = None
        self.qubit_count = None
        self.show_data = []
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None

    def add_data(self, val_curr: list, val_real: list, qubit_count: list):
        '''
        Add data to the metrics class.
        :param val_curr: Current values
        :param val_real: Real values
        :param qubit_count: Number of qubits
        '''

        self.val_curr = val_curr
        self.val_real = val_real
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

    # Draw graphs
    def draw_recall(self, data: list, qubits: list) -> None:
        plt.xlabel('Dimension')
        plt.ylabel('Recall')
        plt.title('Recall of the ' + self.name + ' model')

        plt.bar(qubits, [el[0] for el in data], color='blue', label='Recall')
        plt.legend(loc='lower right')
        plt.show()

    def draw_accuracy(self, data: list, qubits: list) -> None:
        plt.xlabel('Dimension')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of the ' + self.name + ' model')

        plt.bar(qubits, [el[3]*100 for el in data], color='red', label='Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    def draw_rmse(self, data: list, qubits: list) -> None:
        plt.xlabel('Dimension')
        plt.ylabel('RMSE')
        plt.title('RMSE of the ' + self.name + ' model')

        plt.bar(qubits, [el[1] for el in data], color='orange', label='Rmse')
        plt.legend(loc='lower right')
        plt.show()

    def draw_mse(self, data: list, qubits: list) -> None:
        plt.xlabel('Dimension')
        plt.ylabel('MSE')
        plt.title('MSE of the ' + self.name + ' model')

        plt.bar(qubits, [el[2] for el in data], color='brown', label='Mse')
        plt.legend(loc='lower right')
        plt.show()

    def draw_precision(self, data: list, qubits: list) -> None:
        plt.xlabel('Dimension')
        plt.ylabel('Precision')
        plt.title('Precision of the ' + self.name + ' model')

        plt.bar(qubits, [el[4] for el in data], color='grey', label='Precision')
        plt.legend(loc='lower right')
        plt.show()

    def draw_f_measure(self, data: list, qubits: list) -> None:
        plt.xlabel('Dimension')
        plt.ylabel('F-measure')
        plt.title('F-measure of the ' + self.name + ' model')

        plt.bar(qubits, [el[5] for el in data], color='pink', label='F-measure')
        plt.legend(loc='lower right')
        plt.show()


# Test class for Metrics
class MetricsTest:
    def __init__(self):
        pass

    # Calculate and print accuracy, recall, precision, and f-measure for validation data
    def print_metrics(self) -> None:
        print("Recall:", round(self.metrics.recall(), 4))
        print("RMSE:", round(self.metrics.rmse(), 4))
        print("MSE:", round(self.metrics.mse(), 4))
        print("Accuracy:", round(self.metrics.accuracy(), 4))
        print("Precision:", round(self.metrics.precision(), 4))
        print("F-measure:", round(self.metrics.f_measure(), 4))
        print("--------------------")

    def runTest(self) -> None:
        '''
        Run test for Metrics class.
        Test example: Random data for 60 qubits.
        Pipeline:
        1. Create Metrics object.
        2. Add data of current and real values for 60 qubits (dimensions).
        3. Print metrics for each dimension.
        '''

        self.metrics = Metrics("Test")

        for i in range(60, 1, -1):
            self.val_curr = np.random.randint(0, 2, size=10)
            self.val_real = np.random.randint(0, 2, size=10)
            self.metrics.add_data(self.val_curr, self.val_real, i)

        self.show_graphs()

    def show_graphs(self):
        self.metrics.draw_recall(self.metrics.show_data, range(60, 1, -1))
        self.metrics.draw_accuracy(self.metrics.show_data, range(60, 1, -1))
        self.metrics.draw_rmse(self.metrics.show_data, range(60, 1, -1))
        self.metrics.draw_mse(self.metrics.show_data, range(60, 1, -1))
        self.metrics.draw_precision(self.metrics.show_data, range(60, 1, -1))
        self.metrics.draw_f_measure(self.metrics.show_data, range(60, 1, -1))

def double_metrics():
    '''
    Compare metrics of two models. Quantum and Classic.
    Graph show Recall of Quantum and Classic models for each dimension (qubits).
    pipeline:
    1. Create Metrics object for Quantum and Classic models.
    2. Add data of current and real values for 10 qubits.
    3. Print metrics for each dimension.
    '''

    quantum_metrics = Metrics("QSVM")
    classic_metrics = Metrics("SVM")

    # Example of random data
    # val_curr = []
    # val_real = []
    # for i in range(9, 0, -1):
    #     val_curr = np.random.randint(0, 2, size=10)
    #     val_real = np.random.randint(0, 2, size=10)
    #     quantum_metrics.add_data(val_curr, val_real, i)
    #
    # for i in range(9, 0, -1):
    #     val_curr = np.random.randint(0, 2, size=10)
    #     val_real = np.random.randint(0, 2, size=10)
    #     classic_metrics.add_data(val_curr, val_real, i)
    # quantum_recall = [el[0] for el in quantum_metrics.show_data]
    # classic_recall = [el[0] for el in classic_metrics.show_data]

    # Test data
    quantum_recall = [0.484, 0.532, 0.668, 0.636, 0.673, 0.682, 0.690, 0.773, 0.682, 0.727]
    classic_recall = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5]

    n = len(quantum_recall)
    ind = np.arange(n)

    width = 0.35
    plt.bar(ind, quantum_recall, width, color='royalblue', label='Quantum Recall')
    plt.bar(ind + width, classic_recall, width, color='orangered', label='Classic Recall')

    plt.xlabel('Dimension (Qubit)')
    plt.ylabel('Recall')

    plt.legend(loc='lower right')
    plt.show()

def training_run_time_graph():
    '''
    Compare metrics of two models. Quantum and Classic.
    Graph show Time of training of Quantum and Classic models for each dimension (qubits).
    pipeline:
    1. Create Metrics object for Quantum and Classic models.
    2. Add data of training time for 10 qubits.
    3. Print metrics for each dimension.
    '''

    quantum_metrics = Metrics("QSVM")
    classic_metrics = Metrics("SVM")

if __name__ == "__main__":
    testmetrics = MetricsTest()
    testmetrics.runTest()
    double_metrics()
