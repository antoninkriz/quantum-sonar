import numpy as np
import math

# Class to calculate accuracy and recall for validation and training data
class Metrics:
    def __init__(self, val_curr, val_real):
        self.val_curr = np.array(val_curr)
        self.val_real = np.array(val_real)

    def _recall(self, curr, real): # Hard recall
        # Calculate recall
        true_positives = np.sum(np.logical_and(curr >= 0.5, real >= 0.5))
        possible_positives = np.sum(real >= 0.5)
        recall = true_positives / (possible_positives + 1e-9)
        return recall

    def _rmse_recall(self, curr, real): # Soft recall
        rmse = math.sqrt(np.mean((curr - real) ** 2))
        return rmse

    def recall(self):
        return round(self._recall(self.val_curr, self.val_real), 4)

    def rmse_recall(self):
        return 1 - round(self._rmse_recall(self.val_curr, self.val_real), 4)





# (NOT) "Test" class for Metrics
class MetricsTest:
    def __init__(self):
        val_curr = [0, 0.9, 0, 0, 0.5, 0.9, 0.9, 0, 0.9, 0.9]  # predicted validation labels
        val_real = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]  # true validation labels

        self.metrics = Metrics(val_curr, val_real)

    def runTest(self):
        # Calculate and print accuracy and recall for validation and training data
        print("Validation rmse Recall:", self.metrics.rmse_recall())
        print("Validation Recall:", self.metrics.recall())

if __name__ == "__main__":
    test = MetricsTest()
    test.runTest()
