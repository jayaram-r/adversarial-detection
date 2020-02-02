import foolbox
import torch
import torchvision
import numpy as np


def foolbox_attack_helper(attack_model, device, test_loader):
    # model.eval()
    temp = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data_numpy = data.data.cpu().numpy()
        target_numpy = target.data.cpu().numpy()
        shape = data.shape
        adversarials = attack_model(data_numpy, target_numpy, epsilons=1000, max_epsilon=0.5) #, target_numpy, unpack=False)
        if batch_idx == 0:
            total = adversarials
        else:
            #print(total.shape, adversarials.shape)
            total = np.vstack((total, adversarials))
    #print(total.shape)
    return total #numpy ndarray


def foolbox_attack(model, device, loader, bounds, num_classes=10, p_norm='2', adv_attack='FGSM'):
        distance = None
        if p_norm == '2':
            distance = foolbox.distances.Linf
        elif p_norm == 'inf':
            distance = foolbox.distances.Linf
        else:
            raise ValueError("'{}' is not a valid or supported p-norm type".format(args.p_norm))

        model.to(device)
        model.eval()
        fmodel = foolbox.models.PyTorchModel(model, bounds=bounds, num_classes=num_classes)

        print("{} is the attack choice".format(adv_attack))
        # to do: add the parameters for (a) num iterations, and (b) epsilon
        if adv_attack == 'FGSM':
            attack_model = foolbox.attacks.FGSM(fmodel, distance=distance) # distance=foolbox.distances.Linf)
        elif adv_attack == 'PGD':
            attack_model = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel, distance=distance)
        elif adv_attack == 'CW':
            attack_model = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance)
        else:
            raise ValueError("'{}' is not a valid adversarial attack".format(adv_attack))

        #attack_model.as_generator(epsilon=0.2)
        adversarials = foolbox_attack_helper(attack_model, device, loader)
        return adversarials
        # adversarials = attack_model(images, labels)
