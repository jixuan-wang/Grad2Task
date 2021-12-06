################################
# Utilities for model training #
################################

import math
import torch
import torch.nn.functional as F

def get_device_name(device):
    return str(device).replace(':', '_')

def loss(test_logits_sample, test_labels, device=None):
    """
    Compute the cross entropy loss.
    """
    return F.cross_entropy(test_logits_sample, test_labels)

def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    # averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(test_logits_sample, dim=-1)).float())

def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)

def extract_indices(seq, which_one):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    mask = torch.eq(seq, which_one)  # binary mask of labels equal to which_class
    mask_indices = torch.nonzero(mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(mask_indices, (-1,))  # reshape to be a 1D vector

class ValidationAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets, early_stop_steps=5):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
        self.current_best_accuracy_dict = {}
        for dataset in self.datasets:
            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}
        self.not_improved_for = 0
        self.earyly_stop_steps = early_stop_steps

    def is_better(self, accuracies_dict):
        is_better = False
        is_better_count = 0
        for i, dataset in enumerate(self.datasets):
            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
                is_better_count += 1

        if is_better_count >= int(math.floor(self.dataset_count / 2.0)):
            is_better = True
            self.not_improved_for = 0
        else:
            self.not_improved_for += 1

        return is_better
    
    def early_stop(self):
        return self.not_improved_for >= self.earyly_stop_steps

    def replace(self, accuracies_dict):
        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logger, accuracy_dict):
        logger.info("Validation Accuracies:")
        for dataset in self.datasets:
            logger.info("{0:}: {1:.1f}+/-{2:.1f}".format(dataset,
                                                            accuracy_dict[dataset]["accuracy"],
                                                            accuracy_dict[dataset]["confidence"]))
    def get_current_best_accuracy_dict(self):
        return self.current_best_accuracy_dict

class ValidationAccuraciesByAverage(ValidationAccuracies):
    def is_better(self, accuracies_dict):
        is_better = False

        best_sum = sum([self.current_best_accuracy_dict[dataset]["accuracy"] for dataset in self.datasets])
        cur_sum = sum([accuracies_dict[dataset]["accuracy"] for dataset in self.datasets])
        if cur_sum > best_sum:
            is_better = True
            self.not_improved_for = 0
        else:
            self.not_improved_for += 1

        return is_better

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
class EuclideanDist(torch.nn.Module):
    def __init__(self, dim=-1):
        super(EuclideanDist, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        # mutiply by -1 so the result can be used as logits directly
        return -torch.pow(x1 - x2, 2).sum(self.dim)

def dist_metric_by_name(name):
    if name is None or type(name) is not str:
        raise ValueError(f'Invalid distance metric name: {name}')

    if name.lower() == 'cos':
        return torch.nn.CosineSimilarity(dim=-1)
    elif name.lower() == 'euc':
        return EuclideanDist(dim=-1)
    else:
        raise ValueError(f'Invalid distance metric name: {name}')