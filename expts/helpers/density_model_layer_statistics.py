"""
Probability density models for the joint distribution of the test statistics from different layers of a DNN.
The models can be made conditional on either the predicted class or the source class.

We use a multivariate log-normal mixture as the parametric model for the test statistics because they are usually
non-negative valued. This is essentially equivalent to modeling the log of the test statistics using a multivariate
mixture of Gaussian densities. The number of component is set using the bayesian information criterion (BIC) for
model complexity.

"""
import numpy as np
from sklearn.mixture import GaussianMixture
