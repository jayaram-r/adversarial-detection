"""
Implemention of the adversarial attack detection method from the paper:

Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality."
arXiv preprint arXiv:1801.02613 (2018).
https://arxiv.org/pdf/1801.02613.pdf

"""
import numpy as np
import torch
import logging
import copy
from helpers.constants import (
    ROOT,
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST,
    METRIC_DEF,
    FPR_TARGETS_DEF
)
from helpers.lid_estimators import lid_mle_amsaleg, estimate_intrinsic_dimension
from sklearn.linear import LogisticRegression


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
