# Code
Collaborative Learning of Sample Selection and  Robust Functions for Noisy Labels

# Requirements
Python >= 3.8, Pytorch >= 2.0.0, Cuda >= 11.8.0, torchvision >= 0.4.1, mlconfig >= 0.2.0


--exp_name:  folder to save the experimental results.\
--noisy_rate:  noise rate of the dataset.\
--asym:  noise type.\
--seed:  random seed.\
--config_path:  the path of config.\
--version:  the type of loss function.\
--data_type:  name of the dataset.

# How to run
First, you need to unzip the configs.zip file, and place the extracted folder in the same directory as the code. \
Second, please double-check the configs ".yaml" files and the run command before each run.

Here is an example：

python3  main.py --exp_name      test_exp \\ \
                    --noise_rate    0.4                  \\ \
                    --version       nce+rce              \\ \
                    --config_path   configs/cifar10/asym \\ \
                    --seed          123 \\ \
	                  --data_type  cifar10
