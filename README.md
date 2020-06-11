### Detection of adversarial attacks and OOD samples at test-time on deep neural network classifiers
Code and experiments for the anomaly detection paper.

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

### Adversarial Detection
The main script for launching the different adversarial detection methods is `detection_main.py`. To see a description of the command line options, type `python detection_main.py -h`.
A number of these options can be left to their default values and not all of them apply to the different methods.
The usage of this script is summarized below.

#### Running the proposed method, ReBeL
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

### Out-of-distribution Detection
The script for launching the different OOD detection methods is `outlier_detection_main.py`. To see a description of the command line options, type `python outlier_detection_main.py -h`.

### Running the Corrected (Combined) Classification Experiments
The script for launching the corrected classification experiments is `prediction_main.py`. To see a description of the command line options, type `python prediction_main.py -h`.

### Pre-processing and dimensionality reduction of the DNN layer representations
The script `layers.py` can be used for this. To see a description of the command line options, type `python layers.py -h`.
