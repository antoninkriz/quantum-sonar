import numpy as np

# Metrics class to calculate accuracy and recall
class Metrics:
    def __init__(self, val_curr, val_real, train_curr, train_real):
        self.val_curr = np.array(val_curr)
        self.val_real = np.array(val_real)
        self.train_curr = np.array(train_curr)
        self.train_real = np.array(train_real)

    # Recall
    def _recall(self, curr, real):
        true_positives = sum((curr == 1) & (real == 1))
        false_negatives = sum((curr == 0) & (real == 1))

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        return recall

    # Accuracy
    def _accuracy(self, curr, real):
        correct_predictions = sum(curr == real)
        total_predictions = len(real)

        acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        return acc


    # Calculate recall and accuracy for validation and training data
    def val_recall(self):
        return self._recall(self.val_curr, self.val_real)

    def train_recall(self):
        return self._recall(self.train_curr, self.train_real)

    def val_accuracy(self):
        return self._accuracy(self.val_curr, self.val_real)

    def train_accuracy(self):
        return self._accuracy(self.train_curr, self.train_real)

class MetricsTest:

    def __init__(self):
        val_curr = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]  # predicted validation labels
        val_real = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]  # true validation labels
        train_curr = [0, 1, 1, 1, 0, 1, 0, 0, 1, 1]  # predicted training labels
        train_real = [0, 1, 1, 1, 1, 1, 0, 0, 1, 1]  # true training labels

        self.metrics = Metrics(val_curr, val_real, train_curr, train_real)

    def runTest(self):
        # Calculate and print accuracy and recall for validation and training data
        print("Validation Accuracy:", self.metrics.val_accuracy())
        print("Validation Recall:", self.metrics.val_recall())
        print("Training Accuracy:", self.metrics.train_accuracy())
        print("Training Recall:", self.metrics.train_recall())


if __name__ == "__main__":
    test = MetricsTest()
    test.runTest()