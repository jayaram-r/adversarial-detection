### Detection of adversarial attacks and OOD samples at test-time on deep neural network classifiers
Code and experiments for the anomaly detection paper:\
"A General Framework For Detecting Anomalous Inputs to DNN Classifiers", ICML 2021

An earlier version of this paper can be found on arXiv:\
"Detecting Anomalous Inputs to DNN Classifiers By Joint Statistical Testing at the Layers": https://arxiv.org/abs/2007.15147

### Package Requirements
The following packages with their corresponding version should be installed before running the experiments:
- numpy - 1.16.2
- scipy - 1.2.1
- joblib - 0.14.0
- scikit-learn - 0.20.3
- numba - 0.46.0
- llvmlite - 0.30.0
- pynndescent - 0.3.3
- PyTorch - 1.3
- Foolbox - 2.3.0
- matplotlib >= 3.0.3

It is recommended to use a setup a Conda environment with `python3` and install these packages using conda. A specification file for the Conda environment with these packages is provided at `expts/conda_spec_file.txt`.

`PyNNDescent` is an approximate nearest neighbors library that we use, and can be installed from [here](https://github.com/lmcinnes/pynndescent). 
Make sure to install version `0.3.3`, with `numba` version `0.46` and `llvmlite` version `0.30.0`.

### Overview of the Code
The DNN models trained on MNIST, SVHN, and CIFAR-10 datasets can be found under `expts/models/`.

Implementation of all the detection methods can be found under `expts/detectors/`.\
Implementation of the proposed detection method can be found in `expts/detectors/detector_proposed.py`.\
A number of test statistics calculated from the DNN layers have been implemented in `expts/detectors/test_statistics_layers.py`.\
Estimation of p-values at the layers and layer pairs is implemented in `expts/detectors/pvalue_estimation.py`.\
Estimation of multivariate p-value (for a test statistic vector) based on the `aK-LPE` method is implemented in `expts/detectors/localized_pvalue_estimation.py`. \
A number of supporting utilities (e.g. kNN graph construction, intrinsic dimension estimation) have been implemented under `expts/helpers/`.

### Generating Data
The first step prior to running the detection methods is to load the image classification datasets and perform a 5-fold 
cross-validation split of the clean images.
The adversarial attack data are generated from the clean samples in each of the cross-validation folds.
The script `generate_samples.py` can be used both for creating this cross-validation setup and for generating adversarial 
attack data. After running this script, the training and test data should be saved  as numpy files in sub-directories (named `fold1`, `fold2` etc.) corresponding to each dataset (e.g. mnist).


### Pre-processing and dimensionality reduction of the DNN layer representations
The script `layers.py` can be used to perform dimensionality reduction on the layer representations.
It uses a combination of principal components analysis (PCA) and neighborhood preserving projections (NPP), both of which 
learn a linear projection matrix to transform an input vector to a lower dimension.
To see a description of the command line options, type `python layers.py -h`.
The script saves the projection matrix information to a pickle file named `models_dimension_reduction.pkl` in the output 
directory (specified by the option `-o`).
It also saves a summary of the dimensionality reduction to a file named `output_layer_extraction.txt` in the output directory.

### Detecting Adversarial Samples
The main script for launching the different adversarial detection methods is `detection_main.py`. To see a description of the command line options, type `python detection_main.py -h`.
A number of these options can be left to their default values and not all of them apply to the different methods.
Details on the usage of this script are given below.

#### Running the proposed method, JTLA
```bash
#Detection method
dm='proposed'

#Test statistic
#Options are: 'multinomial', 'binomial', 'trust', 'lid', 'distance'
ts='multinomial'

#Scoring method
#Options are: 'pvalue' and 'klpe'
score='pvalue'

#Method for combing p-values from the layers. Applies only if score='pvalue'
#Options are: 'fisher' and 'harmonic_mean'
pf='fisher'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Adversarial attack
#Options are: 'CW', 'PGD', 'FGSM', 'Custom'
attack='CW'

#Output directory
output_dir='./outputs_rebel_fisher'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u detection_main.py -m $dataset --dm $dm --ts $ts --st $score --pf $pf --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

#Change the p-value fusion method to 'harmonic_mean'
pf='harmonic_mean'

#Output directory
output_dir='./outputs_rebel_hmp'

python -u detection_main.py -m $dataset --dm $dm --ts $ts --st $score --pf $pf --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

#Change the scoring method to 'klpe'
score='klpe'

#Output directory
output_dir='./outputs_rebel_klpe'

python -u detection_main.py -m $dataset --dm $dm --ts $ts --st $score --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir
```

#### Running other detection methods
```bash
########################## Deep KNN ###################
dm='dknn'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Adversarial attack
#Options are: 'CW', 'PGD', 'FGSM', 'Custom'
attack='CW'

#Output directory
output_dir='./outputs_dknn'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u detection_main.py -m $dataset --dm $dm --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir


########################## Trust Score ###################
dm='trust'

#Which layer of the DNN to use
#Options are: 'input', 'logit', 'prelogit'
layer='prelogit'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Adversarial attack
#Options are: 'CW', 'PGD', 'FGSM', 'Custom'
attack='CW'

#Output directory
output_dir='./outputs_trust'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u detection_main.py -m $dataset --dm $dm --lts $layer --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir


########################## Deep Mahalanobis detector ###################
dm='mahalanobis'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Adversarial attack
#Options are: 'CW', 'PGD', 'FGSM', 'Custom'
attack='CW'

#Output directory
output_dir='./outputs_mahalanobis'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u detection_main.py -m $dataset --dm $dm --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir


########################## Local intrinsic dimension based detector ###################
dm='lid'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Adversarial attack
#Options are: 'CW', 'PGD', 'FGSM', 'Custom'
attack='CW'

#Output directory
output_dir='./outputs_lid'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u detection_main.py -m $dataset --dm $dm --batch-lid --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir


########################## Odds are odd detector ###################
dm='odds'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Adversarial attack
#Options are: 'CW', 'PGD', 'FGSM', 'Custom'
attack='CW'

#Output directory
output_dir='./outputs_odds'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u detection_main.py -m $dataset --dm $dm --batch-lid --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

```

### Detecting Out-of-distribution Samples 
The script for launching the different OOD detection methods is `outlier_detection_main.py`. To see a description of the command line options, type `python outlier_detection_main.py -h`.
Details on the usage of this script are given below.

#### Running the proposed method, JTLA
```bash
#Detection method
dm='proposed'

#Test statistic
#Options are: 'multinomial', 'binomial', 'trust', 'lid', 'distance'
ts='multinomial'

#Scoring method
#Options are: 'pvalue' and 'klpe'
score='pvalue'

#Method for combing p-values from the layers. Applies only if score='pvalue'
#Options are: 'fisher' and 'harmonic_mean'
pf='fisher'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Output directory
output_dir='./outputs_rebel_fisher'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u outlier_detection_main.py -m $dataset --censor-classes --dm $dm --ts $ts --st $score --pf $pf --gpu $gpu --n-jobs $n_jobs -o $output_dir

#Change the scoring method to 'klpe'
score='klpe'

#Output directory
output_dir='./outputs_rebel_klpe'

python -u outlier_detection_main.py -m $dataset --censor-classes --dm $dm --ts $ts --st $score --gpu $gpu --n-jobs $n_jobs -o $output_dir
```

#### Running other detection methods
```bash
########################## Deep KNN ###################
dm='dknn'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Output directory
output_dir='./outputs_dknn'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u outlier_detection_main.py -m $dataset --censor-classes --dm $dm --gpu $gpu --n-jobs $n_jobs -o $output_dir


########################## Trust Score ###################
dm='trust'

#Which layer of the DNN to use
#Options are: 'input', 'logit', 'prelogit'
layer='prelogit'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Output directory
output_dir='./outputs_trust'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u outlier_detection_main.py -m $dataset --censor-classes --dm $dm --lts $layer --gpu $gpu --n-jobs $n_jobs -o $output_dir


########################## Deep Mahalanobis detector ###################
dm='mahalanobis'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Output directory
output_dir='./outputs_mahalanobis'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

python -u outlier_detection_main.py -m $dataset --dm $dm --censor-classes --gpu $gpu --n-jobs $n_jobs -o $output_dir
```

### Corrected (Combined) Classification Experiments
The script for launching the corrected classification experiments is `prediction_main.py`. To see a description of the command line options, type `python prediction_main.py -h`.

#### Running the proposed method, JTLA
```bash
#Detection method
dm='proposed'

#Test statistic
#Options are: 'multinomial', 'binomial', 'trust', 'lid', 'distance'
ts='multinomial'

#Scoring method
#Options are: 'pvalue' and 'klpe'
score='pvalue'

#Method for combing p-values from the layers. Applies only if score='pvalue'
#Options are: 'fisher' and 'harmonic_mean'
pf='fisher'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Output directory
output_dir='./outputs_rebel_fisher'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

#Adversarial attack
#Set to  'none' to evaluate on clean data
attack='none'

python -u prediction_main.py -m $dataset --dm $dm --ts $ts --st $score --pf $pf --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

#Evaluate on CW attack with confidence = 0
attack='CW'
index=0

python -u prediction_main.py -m $dataset --dm $dm --index-adv $index --ts $ts --st $score --pf $pf --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

```

#### Running deep KNN
```bash
#Detection method
dm='dknn'

#Dataset
#Options are: 'mnist', 'cifar10', 'svhn'
dataset='mnist'

#Output directory
output_dir='./outputs_dknn'

#Number of CPU cores
n_jobs=16

#GPU ID
gpu=0

#Adversarial attack
#Set to  'none' to evaluate on clean data
attack='none'

python -u prediction_main.py -m $dataset --dm $dm --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

#Evaluate on CW attack with confidence = 0
attack='CW'
index=0

python -u prediction_main.py -m $dataset --dm $dm --index-adv $index --adv-attack $attack --gpu $gpu --n-jobs $n_jobs -o $output_dir

```

### Adaptive attack proposed in the paper
Our implementation of the adaptive attack based on the class labels of the k-nearest neighbors of the layer representations 
can be found in `helpers/knn_attack.py`.
The script `generate_samples_custom.py` uses this implementation to generate adversarial samples based on this attack.
