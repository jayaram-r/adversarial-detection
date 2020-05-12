"""
https://github.com/yk/icml19_public
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
        # to do: pending verification
        ##params from torch_example.py
        batch_size=32
        eval_bs=256
        cuda=True
        attack_lr=.25
        eps=8/255
        pgd_iters=10
        # noise parameters
        # prefix 'n' is for Gaussian(0, 1), 'u' is for uniform(-1, 1), 's' is for sign(uniform(-1, 1))
        noise_eps = 'n0.01,s0.01,u0.01,n0.02,s0.02,u0.02,s0.03,n0.03,u0.03'
        noise_eps_detect = 'n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008'
        clip_alignments=True
        debug=False
        #mode='eval'
        wdiff_samples=256
        maxp_cutoff=.999    # used to set the threshold for detection using `scipy.stats.norm.ppf(maxp_cutoff)`
        collect_targeted=False
        save_alignments=False
        load_alignments=False
        fit_classifier=False    # setting to False because we don't need the corrected predictions
        just_detect=True    # we only need the adversarial detections and not the corrected predictions
        
        #initializations
        cuda = cuda and th.cuda.is_available()
        #weights connecting the pre-logit (FC) layer to the logit layer
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


        #function below is defined in detectors/tf_robustify.py
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


def detect_odds_are_odd(predictor, test_loader, adv_loader, model, num_classes, with_attack=True):
        eval_batches=None
        cuda=True
        optim='sgd'
        eps=8/255
        eps_rand=None
        eps_eval=None
        clamp_uniform=False

        #pgd parameters
        load_pgd_test_samples=None

        # initializations
        cuda = cuda and th.cuda.is_available()

        mean_eps = .1
        eps_rand = eps_rand or eps
        eps_eval = eps_eval or eps
        mean_eps = mean_eps * eps_eval

        #variables required for results; not used
        '''
        eval_loss_clean = []
        eval_acc_clean = []
        eval_loss_rand = []
        eval_acc_rand = []
        eval_loss_pgd = []
        eval_acc_pgd = []
        eval_loss_pand = []
        eval_acc_pand = []
        #all_outputs = []
        #diffs_rand, diffs_pgd, diffs_pand = [], [], []
        #eval_preds_clean, eval_preds_rand, eval_preds_pgd, eval_preds_pand = [], [], [], []
        #norms_clean, norms_pgd, norms_rand, norms_pand = [], [], [], []
        #norms_dpgd, norms_drand, norms_dpand = [], [], []
        #eval_important_valid = []
        eval_loss_incr = []
        eval_conf_pgd = []

        #wdiff_corrs = []
        #udiff_corrs = []
        #grad_corrs = []
        #minps_clean = []
        #minps_pgd = []

        acc_clean_after_corr = []
        acc_pgd_after_corr = []
        eval_x_pgd_l0 = []
        eval_x_pgd_l2 = []

        #all_eval_important_pixels = []
        #all_eval_important_single_pixels = []
        #all_eval_losses_per_pixel = []
        '''
        eval_det_clean = []
        eval_det_pgd = []

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
            logits = net_forward(x, layer_by_layer=False) #returns predictions (probability distribution)
            loss = loss_fn(logits, y) #loss based on predictions and true labels
            _, preds = th.max(logits, 1) #single prediction from softmax values
            return loss, preds
        
        #pending modification + verification
        #for eval_batch in tqdm.tqdm(itt.islice(test_loader, eval_batches)):
        
        for test_batch, adv_batch in zip(test_loader, adv_loader):
            x, y = test_batch[0], test_batch[1]
            x_pgd, y_pgd = adv_batch[0], adv_batch[1]
            #this could be any adversarial attack although it is named `x_pgd`
            if cuda:
                x, y = x.cuda(), y.cuda()
                x_pgd, y_pgd = x.cuda(), y.cuda()

            '''
            loss_clean, preds_clean = get_loss_and_preds(x, y)

            #append statements
            eval_loss_clean.append(loss_clean.data.cpu().numpy()) #loss on clean samples
            eval_acc_clean.append((th.eq(preds_clean, y).float()).cpu().numpy()) #accuracy on clean samples
            eval_preds_clean.extend(preds_clean) #predictions on clean samples
            '''
            if with_attack: #set to True
                '''
                #create random samples
                if clamp_uniform: #set to False
                    x_rand = x + th.sign(th.empty_like(x).uniform_(-eps_rand, eps_rand)) * eps_rand
                else:
                    x_rand = x + th.empty_like(x).uniform_(-eps_rand, eps_rand)
                
                loss_rand, preds_rand = get_loss_and_preds(x_rand, y)
                eval_loss_rand.append(loss_rand.data.cpu().numpy()) #loss on random samples
                eval_acc_rand.append((th.eq(preds_rand, y).float()).cpu().numpy()) #accuracy on random samples
                eval_preds_rand.extend(preds_rand) #predictions on random samples

                #attack samples generated outside
                if attack == 'PGD':
                    if not load_pgd_test_samples: #load_pgd_test_samples set to False
                        x_pgd = attack_pgd(x, preds_clean, eps=eps_eval) #change function
                elif attack == 'PGD' and distance == 'l2':
                    x_pgd = attack_pgd(x, preds_clean, eps=eps_eval, l2=True)
                elif attack == 'CW':
                    x_pgd = attack_cw(x, preds_clean)
                elif attack == 'mean':
                    x_pgd = attack_mean(x, preds_clean, eps=eps_eval)

                eval_x_pgd_l0.append(th.max(th.abs((x - x_pgd).view(x.size(0), -1)), -1)[0].detach().cpu().numpy()) #max l0 difference between clean sample and pgd sample
                eval_x_pgd_l2.append(th.norm((x - x_pgd).view(x.size(0), -1), p=2, dim=-1).detach().cpu().numpy()) #l2 difference between clean sample and pgd sample

                loss_pgd, preds_pgd = get_loss_and_preds(x_pgd, y) 

                #append statements
                eval_loss_pgd.append(loss_pgd.data.cpu().numpy()) #loss of pgd samples vs. clean labels
                eval_acc_pgd.append((th.eq(preds_pgd, y).float()).cpu().numpy()) #accuracy of predictions on pgd samples vs. clean labels
                
                #from sklearn
                conf_pgd = confusion_matrix(preds_clean.cpu(), preds_pgd.cpu(), np.arange(num_classes))
                conf_pgd -= np.diag(np.diag(conf_pgd))
                
                #append statements
                eval_conf_pgd.append(conf_pgd)
                eval_preds_pgd.extend(preds_pgd)

                loss_incr = loss_pgd - loss_clean
                
                #append statements
                eval_loss_incr.append(loss_incr.detach().cpu())

                x_pand = x_pgd + th.empty_like(x_pgd).uniform_(-eps_rand, eps_rand)
                loss_pand, preds_pand = get_loss_and_preds(x_pand, y)

                eval_loss_pand.append(loss_pand.data.cpu().numpy())
                eval_acc_pand.append((th.eq(preds_pand, y).float()).cpu().numpy())
                eval_preds_pand.extend(preds_pand)
                '''
                
                #using the predictor
                preds_clean_after_corr, det_clean = predictor.send(x.cpu().numpy()).T
                preds_pgd_after_corr, det_pgd = predictor.send(x_pgd.cpu().numpy()).T

                #append statements
                #acc_clean_after_corr.append(preds_clean_after_corr == y.cpu().numpy())
                #acc_pgd_after_corr.append(preds_pgd_after_corr == y.cpu().numpy())
                eval_det_clean.append(det_clean)
                eval_det_pgd.append(det_pgd)
        
        # `eval_det_clean` and `eval_det_pgd` should be a list of float arrays with the detection scores
        eval_det_clean = np.concatenate(eval_det_clean, axis=0)
        eval_det_pgd = np.concatenate(eval_det_pgd, axis=0)

        return eval_det_clean, eval_det_pgd
