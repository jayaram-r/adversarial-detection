"""
Repurposed code from the Github repo associated with the paper:
Roth, Kevin, Yannic Kilcher, and Thomas Hofmann. "The odds are odd: A statistical test for detecting adversarial
examples." arXiv preprint arXiv:1902.04818 (2019).

Repo: https://github.com/yk/icml19_public
"""
from nets.svhn import *
from nets.cifar10 import *
from nets.mnist import *
from nets.resnet import *
import torch as th
from detectors.tf_robustify import *
import itertools as itt
from sklearn.metrics import confusion_matrix
from helpers.utils import get_samples_as_ndarray


def get_wcls(model, model_type):
    if model_type == 'mnist':
        w_cls = model.fc2.weight
    elif model_type == 'cifar10':
        w_cls = list(model.children())[-1].weight
    elif model_type == 'svhn':
        w_cls = model.fc3.weight
    else:
        raise ValueError("Invalid model type '{}'".format(model_type))

    return w_cls.detach()


# Not used
def return_data(model, test_loader):
    X, Y = get_samples_as_ndarray(test_loader)
    #pgd_train = foolbox_attack(model, device, test_loader, bounds, p_norm=p_norm, adv_attack='PGD') #verify if all parameters are defined
    pgd_train = None #to mirror the ICML'19 repo
    return X, Y, pgd_train


def fit_odds_are_odd(train_inputs, train_adv_inputs, model, model_type, num_classes, clip_min, clip_max):
    ## params from `torch_example.py` in `https://github.com/yk/icml19_public`
    batch_size = 32
    eval_bs = 256
    cuda = True
    attack_lr = .25
    eps = 8./255
    pgd_iters = 10
    # noise parameters
    # prefix 'n' is for Gaussian(0, 1), 'u' is for uniform(-1, 1), 's' is for sign(uniform(-1, 1))
    noise_eps = 'n0.01,s0.01,u0.01,n0.02,s0.02,u0.02,s0.03,n0.03,u0.03'
    noise_eps_detect = 'n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008'
    clip_alignments = True
    debug = False
    # mode = 'eval'
    wdiff_samples = 256
    maxp_cutoff = .999    # used to set the threshold for detection using `scipy.stats.norm.ppf(maxp_cutoff)`
    collect_targeted = False
    save_alignments = False
    load_alignments = False
    fit_classifier = False    # setting to False because we don't need the corrected predictions
    just_detect = True    # we only need the adversarial detections and not the corrected predictions

    # initializations
    cuda = cuda and th.cuda.is_available()
    # weights connecting the pre-logit (FC) layer to the logit layer
    w_cls = get_wcls(model, model_type)

    X, Y = train_inputs[0], train_inputs[1]
    if train_adv_inputs is None:
        pgd_train = None
    else:
        pgd_train = train_adv_inputs[0]

    '''
    if type(loader) != tuple: #if loader is actually a torch data loader
        X, Y, pgd_train = return_data(model, loader)
    else:
        X, Y = loader[0], loader[1]
        pgd_train = None
    '''
    loss_fn = th.nn.CrossEntropyLoss(reduction='none')
    loss_fn_adv = th.nn.CrossEntropyLoss(reduction='none')
    if cuda:
        loss_fn.cuda()
        loss_fn_adv.cuda()


    def net_forward(x, layer_by_layer=False):
        if layer_by_layer:
            return model.layer_wise_odds_are_odd(x)
        else:
            return model(x)

    def latent_and_logits_fn(x):
        lat, log = net_forward(x, layer_by_layer=True)
        lat = lat.reshape(lat.shape[0], -1)
        return lat, log

    def get_loss_and_preds(x, y):
        logits = net_forward(x, layer_by_layer=False)
        loss = loss_fn(logits, y)
        _, preds = th.max(logits, 1)
        return loss, preds


    # function below is defined in detectors/tf_robustify.py
    predictor = collect_statistics(
        X, Y,
        latent_and_logits_fn_th=latent_and_logits_fn,
        nb_classes=num_classes,
        weights=w_cls,
        cuda=cuda,
        debug=debug,
        targeted=collect_targeted,
        noise_eps=noise_eps.split(','),
        noise_eps_detect=noise_eps_detect.split(','),
        num_noise_samples=wdiff_samples,
        batch_size=eval_bs,
        pgd_eps=eps,
        pgd_lr=attack_lr,
        pgd_iters=pgd_iters,
        clip_min=clip_min,
        clip_max=clip_max,
        p_ratio_cutoff=maxp_cutoff,
        #save_alignments_dir=ROOT+'/logs/stats' if save_alignments else None,
        save_alignments_dir=None,
        #load_alignments_dir=os.path.expanduser(ROOT+'/data/advhyp/{}/stats'.format(model_type)) if load_alignments else None,
        load_alignments_dir=None,
        clip_alignments=clip_alignments,
        pgd_train=pgd_train,
        fit_classifier=fit_classifier,
        just_detect=just_detect
    )
    return predictor


def detect_odds_are_odd(predictor, test_loader, adv_loader, use_cuda=True):
    # clean data
    eval_det_clean = []
    # acc_clean_after_corr = []
    for x, y in test_loader:
        if use_cuda:
            x, y = x.cuda(), y.cuda()

        # get the detection scores and corrected predictions
        preds_clean_after_corr, det_clean = predictor.send(x.cpu().numpy()).T

        # acc_clean_after_corr.append(preds_clean_after_corr == y.cpu().numpy())
        eval_det_clean.append(det_clean)

    # adversarial data, not necessarsily PGD attack as the name suggests
    eval_det_pgd = []
    # acc_pgd_after_corr = []
    for x_pgd, y_pgd in adv_loader:
        if use_cuda:
            x_pgd, y_pgd = x_pgd.cuda(), y_pgd.cuda()

        # get the detection scores and corrected predictions
        preds_pgd_after_corr, det_pgd = predictor.send(x_pgd.cpu().numpy()).T

        # acc_pgd_after_corr.append(preds_pgd_after_corr == y_pgd.cpu().numpy())
        eval_det_pgd.append(det_pgd)

    # `eval_det_clean` and `eval_det_pgd` should be a list of float arrays with the detection scores
    eval_det_clean = np.concatenate(eval_det_clean, axis=0)
    eval_det_pgd = np.concatenate(eval_det_pgd, axis=0)

    return eval_det_clean, eval_det_pgd
