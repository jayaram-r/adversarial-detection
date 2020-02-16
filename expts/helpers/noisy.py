import numpy as np
import torch
from helpers.utils import calculate_accuracy
from helpers.constants import SEED_DEFAULT


def get_noisy(model, device, test_loader):
    model.eval()
    
    # len_test_loader = len(test_loader)
    len_data = len(test_loader.dataset)
    accuracy = []
    threshold1 = 0.1/100 #0.1% threshold
    threshold2 = 0.5/100 #0.5% threshold

    # First value of 0 corresponds to the case of no noise
    std_dev_list = [i / 255. for i in range(0, 21, 2)]

    with torch.no_grad():
        for std_dev in std_dev_list:
            correct = 0.
            for batch_idx, (data, target) in enumerate(test_loader):
                # print(data.shape, target.shape, type(data), type(target))
                # shape = tuple(list(data.shape))
                # rand = torch.normal(mean=0., std=std_dev, size=shape)
                rand = std_dev * torch.randn(data.shape)
                data = data + rand
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
            
            cur_accuracy = (100. * correct) / len_data
            accuracy.append(cur_accuracy)
            if len(accuracy) > 1:
                if (accuracy[0] - cur_accuracy) >= threshold1:
                    return std_dev

    # process list to return best std_dev value
    for i in range(1, len(accuracy)):
        if abs(accuracy[0] - accuracy[i]) >= threshold2:
            return std_dev_list[i]

    return None


def get_noise_stdev_helper(model, device, data, labels, stdev_arr, accu_min, n_trials):
    shape = data.shape
    accu_avg = np.zeros_like(stdev_arr)
    i_max = -1
    print("stdev\taccuracy")
    for i, sig in enumerate(stdev_arr):
        # Calculate the average accuracy over `n_trials` random Gaussian noise-perturbed samples with standard
        # deviation `sig`
        for t in range(n_trials):
            noise = np.random.normal(loc=0., scale=sig, size=shape)
            data_noisy = data + noise
            accu_avg[i] = accu_avg[i] + calculate_accuracy(model, device, data=data_noisy, labels=labels)

        accu_avg[i] /= n_trials
        print("{:.8f}\t{:.4f}".format(sig, accu_avg[i]))
        if accu_avg[i] < accu_min:
            i_max = i
            break

    err_msg = None
    if i_max < 0:
        err_msg = ("\nStandard deviation values not large enough to cause the required drop in accuracy. "
                   "Try increasing the standard deviation values.")
    if i_max == 0:
        err_msg = ("\nLowest standard deviation value {:.8f} causes the accuracy to drop to {:.4f}. Try decreasing "
                   "the standard deviation values.".format(stdev_arr[0], accu_avg[0]))

    return accu_avg, i_max, err_msg


def get_noise_stdev(model, device, data, labels, tol_drop=0.2, stdev_range=(1e-5, 1.0), n_trials=100,
                    line_search_size=10, seed=SEED_DEFAULT):
    model.eval()
    np.random.seed(seed)

    # Model accuracy on clean data
    accu_max = calculate_accuracy(model, device, data=data, labels=labels)
    print("Accuracy of the model on clean data = {:.4f}.".format(accu_max))
    accu_min = accu_max - tol_drop

    # Coarse list of standard deviation values; linearly-spaced powers of two
    stdev_arr = np.logspace(np.log2(stdev_range[0]), np.log2(stdev_range[1]), num=line_search_size, base=2)
    print("\nDoing a coarse line search between the standard deviation values: ({:.8f}, {:.8f})".
          format(stdev_arr[0], stdev_arr[-1]))
    accu_avg, i_max, err_msg = get_noise_stdev_helper(model, device, data, labels, stdev_arr, accu_min, n_trials)

    if err_msg is None:
        sig_low = stdev_arr[i_max - 1]
        sig_high = stdev_arr[i_max]
        print("\nDoing a finer line search between the standard deviation values: ({:.8f}, {:.8f})".
              format(sig_low, sig_high))
        stdev_arr = np.logspace(np.log2(sig_low), np.log2(sig_high), num=line_search_size, base=2)
        accu_avg, i_max, err_msg = get_noise_stdev_helper(model, device, data, labels, stdev_arr, accu_min, n_trials)

        if err_msg is None:
            stdev_best = stdev_arr[i_max - 1]
            accu_best = accu_avg[i_max - 1]
            print("Best noise standard deviation = {:.8f}. Corresponding average accuracy = {:.4f}.".
                  format(stdev_best, accu_best))

            return stdev_best
        else:
            print(err_msg)
            return None
    else:
        print(err_msg)
        return None


def create_noisy_samples(loader, std_dev):
    X = np.array([])
    Y = np.array([])
    for batch_idx, (data, target) in enumerate(loader):
        # shape = tuple(list(data.shape))
        # rand = torch.normal(mean=0., std=std_dev, size=shape)
        rand = std_dev * torch.randn(data.shape)
        data = data + rand
        data, target = data.cpu().numpy(), target.cpu().numpy()
        if batch_idx == 0:
            X = data
            Y = target[:, np.newaxis]
        else:
            X = np.vstack((X, data))
            Y = np.vstack((Y, target[:, np.newaxis]))

    return X, Y.ravel()
