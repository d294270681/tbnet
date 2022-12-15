"""TB-Net metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score
from mindspore.nn.metrics import Metric


class AUC(Metric):
    """TB-Net metrics method. Compute model metrics AUC."""

    def __init__(self):
        super(AUC, self).__init__()
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []

    def update(self, *inputs):
        """Update list of predictions and labels."""
        all_predict = inputs[1].asnumpy().flatten().tolist()
        all_label = inputs[2].asnumpy().flatten().tolist()
        self.pred_probs.extend(all_predict)
        self.true_labels.extend(all_label)

    def eval(self):
        """Return AUC score"""
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size is not equal to pred_probs.size()')

        auc = roc_auc_score(self.true_labels, self.pred_probs)

        return auc


class ACC(Metric):
    """TB-Net metrics method. Compute model metrics ACC."""

    def __init__(self):
        super(ACC, self).__init__()
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []

    def update(self, *inputs):
        """Update list of predictions and labels."""
        all_predict = inputs[1].asnumpy().flatten().tolist()
        all_label = inputs[2].asnumpy().flatten().tolist()
        self.pred_probs.extend(all_predict)
        self.true_labels.extend(all_label)

    def eval(self):
        """Return accuracy score"""
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size is not equal to pred_probs.size()')

        predictions = [1 if i >= 0.5 else 0 for i in self.pred_probs]
        acc = np.mean(np.equal(predictions, self.true_labels))

        return acc
