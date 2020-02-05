import foolbox
import torch
import torchvision
import numpy as np
from numpy import linalg as LA
from helpers.constants import ROOT

def calc_norm(src, target, norm):
    if src.shape != target.shape:
        return 0
    n_samples = src.shape[0]
    total = 0
    for i in range(n_samples):
        #print(src[i].shape, target[i].shape)
        val = src[i] - target[i]
        val = val.flatten()
        if norm == 'inf':
            temp = LA.norm(val, np.inf)
        elif norm == '0':
            temp = LA.norm(val, 0)
        elif norm == '2':
            temp = LA.norm(val, 2)
        total = total + temp
    return total/n_samples


def foolbox_attack_helper(attack_model, device, data_loader, loader_type, loader_batch_size, adv_attack, dataset, fold_num, p_norm,
                          stepsize=0.001, confidence=0, epsilon=0.3, max_iterations=1000, iterations=40, max_epsilon=1,
                          labels_req=False):
    # model.eval()
    #parameter string to be written into logs
    param_string = "dataset: "+dataset+" fold_num: "+str(fold_num)+" loader_type: "+loader_type
    param_string += " adv_attack: "+adv_attack+" stepsize: "+str(stepsize)+" confidence: "+str(confidence)+" epsilon: "+str(epsilon)
    param_string += " max_iterations: "+str(max_iterations)+" iterations: "+str(iterations)+" max_epsilon: "+str(max_epsilon)
    
    total = []
    total_labels = []
    failure_list = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        data_numpy = data.data.cpu().numpy()
        target_numpy = target.data.cpu().numpy()
        inp_shape = target_numpy.shape()[1:]
        adversarials = None
        if adv_attack == 'FGSM':
            adversarials = attack_model(data_numpy, target_numpy, max_epsilon=max_epsilon, unpack=False)
        
        elif adv_attack == 'PGD':
            adversarials = attack_model(data_numpy, target_numpy, epsilon=epsilon, stepsize=stepsize,
                    binary_search=False, iterations=iterations, unpack=False)
        
        elif adv_attack == 'CW':
            adversarials = attack_model(data_numpy, target_numpy, confidence=confidence,
                    max_iterations=max_iterations, unpack=False)

        #only store those adv. examples with same shape as input
        adv_examples = np.asarray([a.perturbed for a in adversarials if a.shape == inp_shape])
        
        #store indices of those adv. examples where there is a shape mismatch
        failure_list = [(batch_idx * loader_batch_size) + i for i, a in enumerate(adversarials) if a.shape != inp_shape]
        
        #convert the indices list above to a string, to be entered to the log
        failure_string = ""
        for i, element in enumerate(failure_list):
            if i != len(failure_list) - 1:
                failure_string += str(element)+","
            else:
                failure_string += str(element)+"\n"

        norm_diff = calc_norm(adv_examples, data_numpy, p_norm)
        print("average", p_norm,"-norm difference:", norm_diff)

        #if the norm is poor: (a) the shapes of the adv. examples and src. images don't match, or (b) the generated adv. examples are poor
        if norm_diff < 1e-8:
            log_file = open(ROOT+"/logs/attack_status.txt", "a")
            log_file.write(param_string + "\n")
            log_file.write("adv_examples.shape:" + str(adv_examples.shape)+" source.shape:" + str(data_numpy.shape) + "\n") 
            log_file.write("currently processing batch:" + str(batch_idx) + "\n")
            if len(failure_string) != 0:
                log_file.write("indices of loader where failure occured: " + failure_string)
            else:
                log_file.write("no failure occured; just bad examples \n")
            log_file.write("\n")
            log_file.close()
            continue

        if batch_idx == 0:
            total = adv_examples
        else:
            total = np.vstack((total, adv_examples))

        if labels_req:
            adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
            
            #if the labels of adv. examples is the same as those of the src. labels, then the mismatch = 1
            mismatch = np.mean(adversarial_classes == target_numpy)
            print("label mismatch b/w clean and adversarials:", mismatch)

            #write to log
            if mismatch >= 0.5:
                log_file = open(ROOT+"/logs/attack_status.txt", "a")
                log_file.write(param_string + "\n")
                log_file.write("label mismatch occured, with mismatch = "+str(mismatch)+"\n")
                log_file.write("currently processing batch: " + str(batch_idx) + "\n")
                log_file.write("\n")
                log_file.close()
                continue
            
            #print("label mismatch b/w clean and clean:", np.mean(target_numpy == target_numpy))
            adversarial_classes = adversarial_classes.reshape((-1,1))
            if batch_idx == 0:
                total_labels = adversarial_classes
            else:
                total_labels = np.vstack((total_labels, adversarial_classes))

        print("Finished processing batch id:", batch_idx)

    total_labels = total_labels.reshape((-1,))
    if not labels_req:
        return total, _
    else:
        return total, total_labels


def foolbox_attack(model, device, loader, loader_type, loader_batch_size, bounds, num_classes=10, dataset='cifar10', fold_num=1, p_norm='2', adv_attack='FGSM',
                   stepsize=0.001, confidence=0, epsilon=0.3, max_iterations=1000, iterations=40, max_epsilon=1,
                   labels_req=False):
        
        if p_norm == '2':
            distance = foolbox.distances.MSE
        elif p_norm == 'inf':
            distance = foolbox.distances.Linf
        elif p_norm == '0':
            distance = foolbox.distances.L0
        else:
            raise ValueError("'{}' is not a valid or supported p-norm type".format(args.p_norm))

        model.to(device)
        model.eval()
        fmodel = foolbox.models.PyTorchModel(model, bounds=bounds, num_classes=num_classes)

        print("{} is the attack choice".format(adv_attack))
        print("{} is the distance choice.".format(distance))
        # to do: add the parameters for (a) num iterations, and (b) epsilon
        if adv_attack == 'FGSM':
            attack_model = foolbox.attacks.FGSM(fmodel, distance=distance) # distance=foolbox.distances.Linf)
        elif adv_attack == 'PGD':
            attack_model = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel, distance=distance)
        elif adv_attack == 'CW' and p_norm == '2':
            attack_model = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance)
        else:
            raise ValueError("'{}' is not a valid adversarial attack".format(adv_attack))

        # attack_model.as_generator(epsilon=0.2)
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
        # adversarials = attack_model(images, labels)
