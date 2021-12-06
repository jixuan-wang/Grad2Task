############################################
# MixIn classes for building neural models #
############################################
import torch
import torch.nn.functional as F
from utils import aggregate_accuracy, get_device_name, loss

class CrossEntropyMixin(object):
    def loss(self, test_logits_sample, test_labels, device=None):
        """ Compute the cross entropy loss. """
        return F.cross_entropy(test_logits_sample, test_labels)

class ClassificationAccMixin(object):
    def accuracy_fn(self, test_logits_sample, test_labels):
        """
        Compute classification accuracy.
        """
        # averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
        return torch.mean(torch.eq(test_labels, torch.argmax(test_logits_sample, dim=-1)).float())

class GetDeviceMixin(object):
    def get_device(self):
        return next(self.parameters()).device

    def get_device_name(self):
        return get_device_name(self.get_device())
        
