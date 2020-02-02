"""
Main script for running the adversarial and OOD detection experiments.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from nets.mnist import *
from nets.cifar10 import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import (
    ROOT,
    SEED_DEFAULT,
    CROSS_VAL_SIZE,
    ATTACK_PROPORTION_DEF,
    NORMALIZE_IMAGES
)
from detectors.tf_robustify import collect_statistics
from helpers.utils import (
    load_model_checkpoint,
    save_model_checkpoint
)
from helpers.attacks import *
from detectors.logits_and_latents_identifier import (
    get_samples_as_ndarray,
    latent_and_logits_fn,
    get_wcls,
    return_data
)   # ICML 2019
from detectors.detector_lid_paper import (
    flip,
    get_noisy_samples,
    DetectorLID
)   # ICLR 2018
from detectors.detector_proposed import DetectorLayerStatistics


# Proportion of attack samples from each method when a mixed attack strategy is used at test time.
# The proportions should sum to 1. Note that this is a proportion of the subset of attack samples and not a
# proportion of all the test samples.
MIXED_ATTACK_PROPORTIONS = {
    'FGSM': 0.1,
    'PGD': 0.4,
    'CW': 0.5
}


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='cifar10',
                        help='model type or name of the dataset')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    parser.add_argument('--detection-method', '--dm', choices=['proposed', 'lid', 'odds'],
                        default='proposed', help='detection method to run')
    parser.add_argument('--test-statistic', '--ts', choices=['multinomial', 'lid'], default='multinomial',
                        help='type of test statistic to calculate at the layers for the proposed method')
    parser.add_argument('--ood', action='store_true', default=False,
                        help='Perform OOD detection instead of adversarial (if applicable)')
    parser.add_argument('--include-noise', '--in', action='store_true', default=False,
                        help='Include noisy samples in the evaluation')
    parser.add_argument('--model-dim-reduc', '--mdr', default='',
                        help='Path to the saved dimension reduction model file')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the output and model files')
    parser.add_argument('--detection-mechanism', '-dm', default='odds', help='the detection mechanism to use')
    parser.add_argument('--ckpt', default=False, help='to use checkpoint or not')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='FGSM',
                        help='type of adversarial attack')
    parser.add_argument('--attack-proportion', '--ap', type=float, default=ATTACK_PROPORTION_DEF,
                        help='Proportion of attack samples in the test set (default: {:.2f})'.
                        format(ATTACK_PROPORTION_DEF))
    parser.add_argument('--mixed-attack', '--ma', action='store_true', default=False,
                        help='Use option to enable a mixed attack strategy with multiple methods in '
                             'different proportions')
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')
    parser.add_argument('--gpu', type=str, default="1", help='gpus to execute code on')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.output_dir:
        output_dir = os.path.join(ROOT, 'outputs', args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    apply_dim_reduc = False
    if args.detection_method == 'proposed':
        # Dimension reduction is not applied when the test statistic is LID
        if args.test_statistic == 'multinomial':
            apply_dim_reduc = True

    if apply_dim_reduc:
        if not args.model_dim_reduc:
            model_dim_reduc = os.path.join(ROOT, 'outputs', args.model_type, 'models_dimension_reduction.pkl')


    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_path = os.path.join(ROOT, 'data')

    if args.model_type == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['mnist'])]
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=True, transform=transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs
        )
        model = MNIST().to(device)
        model = load_model_checkpoint(model, args.model_type)
        num_classes = 10

    elif args.model_type == 'cifar10':
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 10
        model = ResNet34().to(device)
        model = load_model_checkpoint(model, args.model_type)

    elif args.model_type == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        testset = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 10
        model = SVHN().to(device)
        model = load_model_checkpoint(model, args.model_type)

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))



    """
    # Stratified cross-validation split
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for ind_tr, ind_te in skf.split(data, labels):
        data_tr = data[ind_tr, :]
        labels_tr = labels[ind_tr]
        data_te = data[ind_te, :]
        labels_te = labels[ind_te]
    """


    if args.detection_mechanism == 'odds':
        # to do: pending verification
        ##params from torch_example.py
        batch_size=32
        eval_bs=256
        cuda=True
        attack_lr=.25
        eps=8/255
        pgd_iters=10
        noise_eps='n0.01,s0.01,u0.01,n0.02,s0.02,u0.02,s0.03,n0.03,u0.03'
        noise_eps_detect='n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008'
        clip_alignments=True
        debug=False
        #mode='eval'
        wdiff_samples=256
        maxp_cutoff=.999
        collect_targeted=False
        save_alignments=False
        load_alignments=False
        fit_classifier=True
        just_detect=False
        ######

        #functions below are defined in detectors/logits_and_latents_identifier.py
        w_cls = get_wcls(args.model_type) 
        X, Y, pgd_train = return_data(test_loader)

        #function below is defined in helpers/tf_robustify.py
        predictor = collect_statistics(
            X, Y, latent_and_logits_fn_th=latent_and_logits_fn, nb_classes=num_classes,
            weights=w_cls, cuda=cuda, debug=debug, targeted=collect_targeted, noise_eps=noise_eps.split(','),
            noise_eps_detect=noise_eps_detect.split(','), num_noise_samples=wdiff_samples, batch_size=eval_bs,
            pgd_eps=eps, pgd_lr=attack_lr, pgd_iters=pgd_iters, clip_min=clip_min, clip_max=clip_max,
            p_ratio_cutoff=maxp_cutoff,
            save_alignments_dir=ROOT+'/logs/stats' if save_alignments else None,
            load_alignments_dir=os.path.expanduser(ROOT+'/data/advhyp/{}/stats'.format(args.model_type)) if load_alignments else None,
            clip_alignments=clip_alignments, pgd_train=pgd_train, fit_classifier=fit_classifier,
            just_detect=just_detect
        )

        #pending modification + verification
        for eval_batch in tqdm.tqdm(itt.islice(test_loader, args.eval_batches)):
            if args.load_pgd_test_samples:
                x, y, x_pgd = eval_batch
            else:
                x, y = eval_batch
            if args.cuda:
                x, y = x.cuda(), y.cuda()
                if args.load_pgd_test_samples:
                    x_pgd = x_pgd.cuda()

            loss_clean, preds_clean = get_loss_and_preds(x, y)

            eval_loss_clean.append((loss_clean.data).cpu().numpy())
            eval_acc_clean.append((th.eq(preds_clean, y).float()).cpu().numpy())
            eval_preds_clean.extend(preds_clean)

            if with_attack:
                if args.clamp_uniform:
                    x_rand = x + th.sign(th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)) * args.eps_rand
                else:
                    x_rand = x + th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)
                loss_rand, preds_rand = get_loss_and_preds(x_rand, y)

                eval_loss_rand.append((loss_rand.data).cpu().numpy())
                eval_acc_rand.append((th.eq(preds_rand, y).float()).cpu().numpy())
                eval_preds_rand.extend(preds_rand)

                if args.attack == 'pgd':
                    if not args.load_pgd_test_samples:
                        x_pgd = attack_pgd(x, preds_clean, eps=args.eps_eval)
                elif args.attack == 'pgdl2':
                    x_pgd = attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
                elif args.attack == 'cw':
                    x_pgd = attack_cw(x, preds_clean)

                elif args.attack == 'mean':
                    x_pgd = attack_mean(x, preds_clean, eps=args.eps_eval)
                eval_x_pgd_l0.append(th.max(th.abs((x - x_pgd).view(x.size(0), -1)), -1)[0].detach().cpu().numpy())
                eval_x_pgd_l2.append(th.norm((x - x_pgd).view(x.size(0), -1), p=2, dim=-1).detach().cpu().numpy())

                loss_pgd, preds_pgd = get_loss_and_preds(x_pgd, y)

                eval_loss_pgd.append((loss_pgd.data).cpu().numpy())
                eval_acc_pgd.append((th.eq(preds_pgd, y).float()).cpu().numpy())
                conf_pgd = confusion_matrix(preds_clean.cpu(), preds_pgd.cpu(), np.arange(nb_classes))
                conf_pgd -= np.diag(np.diag(conf_pgd))
                eval_conf_pgd.append(conf_pgd)
                eval_preds_pgd.extend(preds_pgd)

                loss_incr = loss_pgd - loss_clean
                eval_loss_incr.append(loss_incr.detach().cpu())

                x_pand = x_pgd + th.empty_like(x_pgd).uniform_(-args.eps_rand, args.eps_rand)
                loss_pand, preds_pand = get_loss_and_preds(x_pand, y)

                eval_loss_pand.append((loss_pand.data).cpu().numpy())
                eval_acc_pand.append((th.eq(preds_pand, y).float()).cpu().numpy())
                eval_preds_pand.extend(preds_pand)

                preds_clean_after_corr, det_clean = predictor.send(x.cpu().numpy()).T
                preds_pgd_after_corr, det_pgd = predictor.send(x_pgd.cpu().numpy()).T

                acc_clean_after_corr.append(preds_clean_after_corr == y.cpu().numpy())
                acc_pgd_after_corr.append(preds_pgd_after_corr == y.cpu().numpy())

                eval_det_clean.append(det_clean)
                eval_det_pgd.append(det_pgd)



    elif args.detection_mechanism == 'lid':
        # to do: jayaram will complete the procedure
        # required methods are in `detectors.detector_lid_paper`

    elif args.detection_mechanism == 'proposed':
        # to do: jayaram will complete the procedure
        # required methods are in `detectors.detector_proposed`


if __name__ == '__main__':
    main()
