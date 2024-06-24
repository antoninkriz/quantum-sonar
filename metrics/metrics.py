import numpy as np
import math

# Class to calculate accuracy and recall for validation and training data
class Metrics:
    def __init__(self, val_curr, val_real):
        self.val_curr = np.array(val_curr)
        self.val_real = np.array(val_real)

    def _recall(self, curr, real):
        # Calculate recall
        true_positives = np.sum(np.logical_and(curr >= 0.5, real >= 0.5))
        possible_positives = np.sum(real >= 0.5)
        recall = true_positives / (possible_positives + 1e-9)
        return recall

    def _rmse_recall(self, curr, real):
        return math.sqrt(sum((curr - real) ** 2) / len(real))

    def _accuracy(self, curr, real):
        # Calculate accuracy
        correct = np.sum(curr == real)
        total = len(real)
        acc = correct / total
        return acc

    def val_recall(self):
        return self._recall(self.val_curr, self.val_real)

    def val_rmse_recall(self):
        return self._rmse_recall(self.val_curr, self.val_real)

    def val_accuracy(self):
        return self._accuracy(self.val_curr, self.val_real)





class MetricsTest:
    def __init__(self):
        val_curr = [0, 0.2, 0.1, 0, 0.5, 0.3, 0, 0, 0.9, 0]  # predicted validation labels
        val_real = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]  # true validation labels

        self.metrics = Metrics(val_curr, val_real)

    def runTest(self):
        # Calculate and print accuracy and recall for validation and training data
        print("Validation rmse Recall:", self.metrics.val_rmse_recall())
        print("Validation Recall:", self.metrics.val_recall())

if __name__ == "__main__":
    test = MetricsTest()
    test.runTest()
