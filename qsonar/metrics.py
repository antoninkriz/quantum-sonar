import numpy as np

# Class to calculate accuracy, recall, precision, and f-measure for validation and training data
class Metrics:
    def __init__(self, val_curr, val_real):
        self.val_curr = np.array(val_curr)
        self.val_real = np.array(val_real)

        # Calculate TP, TN, FP, FN
        self.TP = np.sum((self.val_curr == 1) & (self.val_real == 1))
        self.TN = np.sum((self.val_curr == 0) & (self.val_real == 0))
        self.FP = np.sum((self.val_curr == 1) & (self.val_real == 0))
        self.FN = np.sum((self.val_curr == 0) & (self.val_real == 1))

    # Recall (Sensitivity)
    def recall(self) -> float:
        if self.TP + self.FN == 0:
            return 0.0
        return self.TP / (self.TP + self.FN)

    def rmse(self) -> float:
        if self.TP + self.FN == 0:
            return 0.0
        return np.sqrt((self.TP / (self.TP + self.FN)) * (1 - (self.TP / (self.TP + self.FN))) / (self.TP + self.FN))

    def mse(self) -> float:
        if self.TP + self.FN == 0:
            return 0.0
        return (self.TP / (self.TP + self.FN)) * (1 - (self.TP / (self.TP + self.FN))) / (self.TP + self.FN)

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

# Test class for Metrics
class MetricsTest:
    def __init__(self):
        val_curr = [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]  # predicted validation labels
        val_real = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]  # true validation labels
        self.metrics = Metrics(val_curr, val_real)

    # Calculate and print accuracy, recall, precision, and f-measure for validation data
    def runTest(self) -> None:
        print("Recall:", round(self.metrics.recall(), 4))
        print("RMSE:", round(self.metrics.rmse(), 4))
        print("MSE:", round(self.metrics.mse(), 4))
        print("Accuracy:", round(self.metrics.accuracy(), 4))
        print("Precision:", round(self.metrics.precision(), 4))
        print("F-measure:", round(self.metrics.f_measure(), 4))

if __name__ == "__main__":
    test = MetricsTest()
    test.runTest()
