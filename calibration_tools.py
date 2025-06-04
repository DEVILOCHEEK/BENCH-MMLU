import numpy as np

def one_hot_encode(labels, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(labels))
    return np.eye(num_classes)[labels]

def get_adaptive_bins(probs, num_bins):
    if len(probs) == 0:
        return np.linspace(0, 1, num_bins+1)[1:]
    edges = np.percentile(probs, np.linspace(0, 100, num_bins+1)[1:-1])
    return edges

def binary_converter(probs):
    return np.array([[1-p, p] for p in probs])

class GeneralCalibrationError:
    def __init__(self, binning_scheme='even', max_prob=True, class_conditional=False,
                 norm='l1', num_bins=10, threshold=0.0, datapoints_per_bin=None):
        self.binning_scheme = binning_scheme
        self.max_prob = max_prob
        self.class_conditional = class_conditional
        self.norm = norm
        self.num_bins = num_bins
        self.threshold = threshold
        self.datapoints_per_bin = datapoints_per_bin
        self.calibration_error = None

    def get_calibration_error(self, probs, labels, bins):
        bin_indices = np.digitize(probs, bins)
        errors = []
        total_count = len(probs)
        for i in range(len(bins)+1):
            idx = bin_indices == i
            if np.sum(idx) == 0:
                continue
            conf = np.mean(probs[idx])
            acc = np.mean(labels[idx])
            error = np.abs(conf - acc) if self.norm == 'l1' else (conf - acc)**2
            weight = np.sum(idx) / total_count
            errors.append(weight * error)
        if self.norm == 'l1':
            return np.sum(errors)
        else:
            return np.sqrt(np.sum(errors))

    def update_state(self, labels, probs):
        probs = np.array(probs)
        labels = np.array(labels)

        if probs.ndim == 1:
            probs = binary_converter(probs)
        
        num_classes = probs.shape[1]
        labels_one_hot = one_hot_encode(labels, num_classes)

        if self.max_prob:
            max_indices = np.argmax(probs, axis=1)
            probs = probs[np.arange(len(probs)), max_indices]
            labels_one_hot = labels_one_hot[np.arange(len(labels_one_hot)), max_indices]
        else:
            probs = probs.flatten()
            labels_one_hot = labels_one_hot.flatten()

        mask = probs > self.threshold
        probs = probs[mask]
        labels_one_hot = labels_one_hot[mask]

        if self.binning_scheme == 'even':
            bins = np.linspace(0, 1, self.num_bins + 1)[1:-1]
        else:
            bins = get_adaptive_bins(probs, self.num_bins)

        self.calibration_error = self.get_calibration_error(probs, labels_one_hot, bins)

    def result(self):
        return self.calibration_error


def gce(labels, probs, **kwargs):
    gce_metric = GeneralCalibrationError(**kwargs)
    gce_metric.update_state(labels, probs)
    return gce_metric.result()


# Приклад використання:
labels = np.array([0, 1, 1, 0, 1])
probs = np.array([0.1, 0.9, 0.8, 0.3, 0.7])

error = gce(labels, probs, binning_scheme='even', max_prob=True, class_conditional=False, norm='l1', num_bins=5)
print("Calibration error:", error)
