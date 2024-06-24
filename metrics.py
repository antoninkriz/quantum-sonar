import numpy as np

class Metrics:
    def __init__(self, val_curr, val_real, train_curr, train_real):
        self.val_curr = np.array(val_curr)
        self.val_real = np.array(val_real)
        self.train_curr = np.array(train_curr)
        self.train_real = np.array(train_real)

    def _recall(self, curr, real):
        true_positives = sum((curr == 1) & (real == 1))
        false_negatives = sum((curr == 0) & (real == 1))

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        return recall

    def _accuracy(self, curr, real):
        correct_predictions = sum(curr == real)
        total_predictions = len(real)

        acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        return acc

    def val_recall(self):
        return self._recall(self.val_curr, self.val_real)

    def train_recall(self):
        return self._recall(self.train_curr, self.train_real)

    def val_accuracy(self):
        return self._accuracy(self.val_curr, self.val_real)

    def train_accuracy(self):
        return self._accuracy(self.train_curr, self.train_real)

# Sample test data
val_curr = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]  # predicted validation labels
val_real = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]  # true validation labels
train_curr = [0, 1, 1, 1, 0, 1, 0, 0, 1, 1]  # predicted training labels
train_real = [0, 1, 1, 1, 1, 1, 0, 0, 1, 1]  # true training labels

# Create an instance of Metrics with test data
metrics = Metrics(val_curr, val_real, train_curr, train_real)

# Calculate and print accuracy and recall for validation and training data
print("Validation Accuracy:", metrics.val_accuracy())
print("Validation Recall:", metrics.val_recall())
print("Training Accuracy:", metrics.train_accuracy())
print("Training Recall:", metrics.train_recall())
