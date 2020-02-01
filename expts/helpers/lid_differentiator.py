import numpy as np

STDEVS = {
    'mnist': {'fgsm': 0.271, 'bim-a': 0.111, 'bim-b': 0.167, 'cw-l2': 0.207},
    'cifar10': {'fgsm': 0.0504, 'bim-a': 0.0084, 'bim-b': 0.0428, 'cw-l2': 0.007},
    'svhn': {'fgsm': 0.133, 'bim-a': 0.0155, 'bim-b': 0.095, 'cw-l2': 0.008}
}

CLIP_MIN = -0.5
CLIP_MAX = 0.5

#PATH_DATA = "data/"


def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < CLIP_MAX)[0]
    #assert candidate_inds.shape[0] >= nb_diff
    if candidate_inds.shape[0] < nb_diff:
        return None
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = CLIP_MAX

    return np.reshape(x, original_shape)


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw-l0']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    else:
        warnings.warn("Important: using pre-set Gaussian scale sizes to craft noisy "
                      "samples. You will definitely need to manually tune the scale "
                      "according to the L2 print below, otherwise the result "
                      "will inaccurate. In future scale sizes will be inferred "
                      "automatically. For now, manually tune the scales around "
                      "mnist: L2/20.0, cifar: L2/54.0, svhn: L2/60.0")
        # Add Gaussian noise to the samples
        # print(STDEVS[dataset][attack])
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                CLIP_MIN
            ),
            CLIP_MAX
        )

    return X_test_noisy

