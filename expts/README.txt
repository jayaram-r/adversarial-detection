1. Create a conda environment based on the instructions in https://github.com/locuslab/smoothing; call it detection
2. We recommend using python3 since the code has not been tested for python2
3. Activate the conda environment detection
4. Install foolbox based on instructions in https://github.com/bethgelab/foolbox ; use the pip based installation
5. /u/c/h/chandrasekaran/anaconda3/envs/detection/lib/python3.7/site-packages/foolbox/adversarial.py --> assertion statement (line 37/38) has been commented
5. /u/c/h/chandrasekaran/anaconda3/envs/detection/lib/python3.7/site-packages/foolbox/attacks/carlini_wagner.py --> made changes to the algorithm to potentially increase speed
