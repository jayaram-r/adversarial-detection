import foolbox
import torch
import torchvision
import numpy as np
import os
from numpy import linalg as LA
from helpers.constants import ROOT


def calc_norm(src, target, norm):
    n_samples = src.shape[0]
    if n_samples == 0:
        return 0.

    if src.shape != target.shape:
        raise Valuerror("Shape mismatch in the inputs")

    # Flatten all but the first dimension and take the vector norm of each row in `diff`
    diff = src.reshape(n_samples, -1) - target.reshape(n_samples, -1)
    if norm == 'inf':
        temp = LA.norm(diff, ord=np.inf, axis=1)
    elif norm == '0':
        temp = LA.norm(diff, ord=0, axis=1)
    elif norm == '1':
        temp = LA.norm(diff, ord=1, axis=1)
    elif norm == '2':
        temp = LA.norm(diff, ord=2, axis=1)
    else:
        raise ValueError("Invalid or unsupported norm type.")

    return np.mean(temp)


def foolbox_attack_helper(attack_model, device, data_loader, loader_type, loader_batch_size, adv_attack, dataset,
                          fold_num, p_norm, stepsize=0.001, confidence=0, epsilon=0.3, max_iterations=1000,
                          iterations=40, max_epsilon=1, min_norm_diff=1e-8):
    # model.eval()
    # parameter string to be written into logs
    param_string = "dataset: " + dataset + " fold_num: " + str(fold_num) + " loader_type: " + loader_type
    param_string += (" adv_attack: " + adv_attack + " stepsize: " + str(stepsize) + " confidence: " + str(confidence)
                     + " epsilon: "+str(epsilon))
    param_string += (" max_iterations: " + str(max_iterations) + " iterations: " + str(iterations) + " max_epsilon: "
                     + str(max_epsilon))

    log_filename = os.path.join(ROOT, 'logs', 'attack_status.txt')
    data_adver = np.array([])
    targets_adver = np.array([])
    data_clean = np.array([])
    targets_clean = np.array([])
    init_batch = True
    n_samples_tot = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        data_numpy = data.data.cpu().numpy()
        target_numpy = target.data.cpu().numpy()
        inp_shape = data_numpy.shape[1:]
        n_samples_tot += target_numpy.shape[0]
        
        adversarials = []
        if adv_attack == 'FGSM':
            adversarials = attack_model(data_numpy, target_numpy, max_epsilon=max_epsilon, unpack=False)
        
        elif adv_attack == 'PGD':
            adversarials = attack_model(data_numpy, target_numpy, epsilon=epsilon, stepsize=stepsize,
                                        binary_search=False, iterations=iterations, unpack=False)
        
        elif adv_attack == 'CW':
            adversarials = attack_model(data_numpy, target_numpy, confidence=confidence,
                                        max_iterations=max_iterations, unpack=False)
        else:
            raise ValueError("'{}' is not a supported adversarial attack".format(adv_attack))

        #only store those adv. examples with same shape as input
        mask_valid = np.array([isinstance(a.perturbed, np.ndarray) and (a.perturbed.shape == inp_shape)
                               for a in adversarials], dtype=np.bool)
        num_adversarials = mask_valid.shape[0]
        if num_adversarials == 0:
            print("\nNo adversarial samples generated from batch {:d}.".format(batch_idx))
            log_file = open(log_filename, "a")
            log_file.write("No adversarial samples generated from batch {:d}.\n\n".format(batch_idx))
            log_file.close()
            continue

        # valid adversarial samples and labels
        num_valid = mask_valid[mask_valid].shape[0]
        adv_examples = np.asarray([a.perturbed for i, a in enumerate(adversarials) if mask_valid[i]])
        adversarial_classes = np.asarray([a.adversarial_class for i, a in enumerate(adversarials)
                                          if mask_valid[i]])
        # clean samples corresponding to the valid adversarial samples
        data_numpy_valid = data_numpy[mask_valid, :]
        target_numpy_valid = target_numpy[mask_valid]
        
        #store indices of those adv. examples where there is a shape mismatch
        failure_list = [str(batch_idx * loader_batch_size + i) for i, a in enumerate(adversarials)
                        if not mask_valid[i]]
        norm_diff = calc_norm(adv_examples, data_numpy_valid, p_norm)
        if num_valid > 0:
            print("average", p_norm, "-norm difference:", norm_diff)

        #if the norm is poor: (a) the shapes of the adv. examples and src. images don't match, or (b) the generated
        # adv. examples are poor
        skip = False
        if norm_diff < min_norm_diff or bool(failure_list):
            log_file = open(log_filename, "a")
            log_file.write(param_string + "\n")
            log_file.write("adv_examples.shape:" + str(adv_examples.shape) + " source.shape:" + str(data_numpy.shape)
                           + "\n")
            log_file.write("currently processing batch:" + str(batch_idx) + "\n")
            if failure_list:
                log_file.write("Indices of loader where failure occured: " + ','.join(failure_list) + '\n')

            if num_valid == 0:
                skip = True
                log_file.write("No valid adversarial samples generated. Skipping this batch.\n")

            if norm_diff < min_norm_diff:
                skip = True
                if num_valid > 0:
                    log_file.write("Average norm difference between the adversarial and original samples is very "
                                   "low. Skipping this batch.\n")

            log_file.write("\n")
            log_file.close()
            if skip:
                continue

        #if the labels of adv. examples is the same as those of the src. labels, then the mismatch = 0
        mismatch = np.mean(adversarial_classes != target_numpy_valid)
        print("label mismatch b/w clean and adversarials:", mismatch)   # should be close to 1 ideally
        #write to log
        if mismatch < 0.5:
            log_file = open(log_filename, "a")
            log_file.write(param_string + "\n")
            log_file.write("currently processing batch: " + str(batch_idx) + "\n")
            log_file.write("Skipping this batch because less than 50% of the adversarial samples have a label "
                           "mismatch with the clean samples. Label mismatch = " + str(mismatch) + "\n\n")
            log_file.close()
            continue

        # Accumulate the results from this batch
        if init_batch:
            data_adver = adv_examples
            targets_adver = adversarial_classes[:, np.newaxis]
            data_clean = data_numpy_valid
            targets_clean = target_numpy_valid[:, np.newaxis]
            init_batch = False
        else:
            data_adver = np.vstack((data_adver, adv_examples))
            targets_adver = np.vstack((targets_adver, adversarial_classes[:, np.newaxis]))
            data_clean = np.vstack((data_clean, data_numpy_valid))
            targets_clean = np.vstack((targets_clean, target_numpy_valid[:, np.newaxis]))

        print("Finished processing batch id:", batch_idx)

    print("Generated {:d} adversarial samples from a total of {:d} clean samples.".format(targets_adver.shape[0],
                                                                                          n_samples_tot))
    assert (data_adver.shape[0] == targets_adver.shape[0] == data_clean.shape[0] == targets_clean.shape[0]), \
        "Number of samples (rows) are not equal in the returned data and label arrays."
    return data_adver, targets_adver.ravel(), data_clean, targets_clean.ravel()


def foolbox_attack(model, device, loader, loader_type, loader_batch_size, bounds, num_classes=10, dataset='cifar10',
                   fold_num=1, p_norm='2', adv_attack='FGSM', stepsize=0.001, confidence=0, epsilon=0.3,
                   max_iterations=1000, iterations=40, max_epsilon=1):

    if p_norm == '2':
        distance = foolbox.distances.MeanSquaredDistance
    elif p_norm == 'inf':
        distance = foolbox.distances.Linf
    elif p_norm == '0':
        distance = foolbox.distances.L0
    else:
        raise ValueError("'{}' is not a valid or supported p-norm type".format(args.p_norm))

    model.to(device)
    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=bounds, num_classes=num_classes)

    print("{} is the distance choice.".format(distance))
    if adv_attack == 'FGSM':
        print("{} is the attack choice".format(adv_attack))
        attack_model = foolbox.attacks.FGSM(fmodel, distance=distance) # distance=foolbox.distances.Linf)
        print(attack_model)
    elif adv_attack == 'PGD' and p_norm == 'inf':
        print("{} is the attack choice".format(adv_attack))
        attack_model = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel, distance=distance)
        print(attack_model)
    elif adv_attack == 'CW' and p_norm == '2':
        print("{} is the attack choice".format(adv_attack))
        attack_model = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance)
        print(attack_model)
    else:
        raise ValueError("'{}' is not a supported adversarial attack".format(adv_attack))

    data_adver, labels_adver, data_clean, labels_clean = foolbox_attack_helper(
        attack_model,
        device,
        loader,
        loader_type,
        loader_batch_size,
        adv_attack,
        dataset,
        fold_num,
        p_norm=p_norm,
        stepsize=stepsize,
        confidence=confidence,
        epsilon=epsilon,
        max_iterations=max_iterations,
        iterations=iterations,
        max_epsilon=max_epsilon
    )
    return data_adver, labels_adver, data_clean, labels_clean
