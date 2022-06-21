from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss, MultiSimilarityLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy, CrossEntropyLabelWeight

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'MultiSimilarityLoss',
    'CrossEntropyLabelWeight'
]
