import foolbox
import torch
import torchvision
import numpy as np
import os
from numpy import linalg as LA
from helpers.constants import ROOT


def calc_norm(src, target, norm):
    if src.shape != target.shape:
        print("ERROR: shape mismatch in the inputs.")
        return 0

    n_samples = src.shape[0]
    total = 0.
    for i in range(n_samples):
        #print(src[i].shape, target[i].shape)
        val = (src[i] - target[i]).flatten()
        temp = 0.
        if norm == 'inf':
            temp = LA.norm(val, np.inf)
        elif norm == '0':
            temp = LA.norm(val, 0)
        elif norm == '2':
            temp = LA.norm(val, 2)

        total += temp

    return total / n_samples


def foolbox_attack_helper(attack_model, device, data_loader, loader_type, loader_batch_size, adv_attack, dataset,
                          fold_num, p_norm, stepsize=0.001, confidence=0, epsilon=0.3, max_iterations=1000,
                          iterations=40, max_epsilon=1, labels_req=False, min_norm_diff=1e-8):
    # model.eval()
    # parameter string to be written into logs
    param_string = "dataset: " +dataset + " fold_num: " + str(fold_num) + " loader_type: " + loader_type
    param_string += (" adv_attack: " + adv_attack + " stepsize: " + str(stepsize) + " confidence: " + str(confidence)
                     + " epsilon: "+str(epsilon))
    param_string += (" max_iterations: " + str(max_iterations) + " iterations: " + str(iterations) + " max_epsilon: "
                     + str(max_epsilon))

    log_filename = os.path.join(ROOT, 'logs', 'attack_status.txt')
    total = np.array([])
    total_labels = np.array([])
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        data_numpy = data.data.cpu().numpy()
        target_numpy = target.data.cpu().numpy()
        inp_shape = data_numpy.shape[1:]
        
        adversarials = []
        if adv_attack == 'FGSM':
            adversarials = attack_model(data_numpy, target_numpy, max_epsilon=max_epsilon, unpack=False)
        
        elif adv_attack == 'PGD':
            adversarials = attack_model(data_numpy, target_numpy, epsilon=epsilon, stepsize=stepsize,
                    binary_search=False, iterations=iterations, unpack=False)
        
        elif adv_attack == 'CW':
            adversarials = attack_model(data_numpy, target_numpy, confidence=confidence,
                    max_iterations=max_iterations, unpack=False)

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

        # valid adversarial samples
        num_valid = mask_valid[mask_valid].shape[0]
        adv_examples = np.asarray([a.perturbed for i, a in enumerate(adversarials) if mask_valid[i]])
        
        #store indices of those adv. examples where there is a shape mismatch
        failure_list = [str(batch_idx * loader_batch_size + i) for i, a in enumerate(adversarials)
                        if not mask_valid[i]]

        norm_diff = calc_norm(adv_examples, data_numpy[mask_valid, :], p_norm)
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
                log_file.write("Average norm difference between the adversarial and original samples is very low. "
                               "Skipping this batch.\n")

            log_file.write("\n")
            log_file.close()
            if skip:
                continue

        if labels_req:
            adversarial_classes = np.asarray([a.adversarial_class for i, a in enumerate(adversarials)
                                              if mask_valid[i]])
            #if the labels of adv. examples is the same as those of the src. labels, then the mismatch = 0
            mismatch = np.mean(adversarial_classes != target_numpy[mask_valid])
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

            if batch_idx == 0:
                total_labels = adversarial_classes[:, np.newaxis]
            else:
                total_labels = np.vstack((total_labels, adversarial_classes[:, np.newaxis]))

        # Include the current batch of adversarial samples only if the corresponding labels have been included
        if batch_idx == 0:
            total = adv_examples
        else:
            total = np.vstack((total, adv_examples))

        print("Finished processing batch id:", batch_idx)

    if not labels_req:
        return total, np.array([])
    else:
        return total, total_labels.ravel()


def foolbox_attack(model, device, loader, loader_type, loader_batch_size, bounds, num_classes=10, dataset='cifar10',
                   fold_num=1, p_norm='2', adv_attack='FGSM', stepsize=0.001, confidence=0, epsilon=0.3,
                   max_iterations=1000, iterations=40, max_epsilon=1, labels_req=False):

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

    adversarials, adversarial_labels = foolbox_attack_helper(
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
        max_epsilon=max_epsilon,
        labels_req=labels_req)

    return adversarials, adversarial_labels
