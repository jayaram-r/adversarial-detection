### Detection of adversarial attacks at test-time on deep neural network classifiers
Code and experiments for the adversarial detection paper.

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

It is recommended to use a setup a Conda environment with `python3` and install these packages using conda.

`PyNNDescent` is an approximate nearest neighbors library that we use, and can be installed from [here](https://github.com/lmcinnes/pynndescent). 
Make sure to install version `0.3.3`, with `numba` version `0.46` and `llvmlite` version `0.30.0`.
