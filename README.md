# SNSSL
Paper: Exploiting Superpixel-based Contextual Information on Active Learning for High Spatial Resolution Remote Sensing Image Classification

Before running, please use "pip install -r requirements.txt" to install the dependencies

Run "train_floodnet.py" or "train_potsdam.py" to start active learning training

When running "train_floodnet.py" or "train_potsdam.py" for the first time, please set use_Xcache=False and check all paths

After saving the classifier model parameters, you can get the results faster with "SNSSL_floodnet.py" or "SNSSL_potsdam.py"

The labels needed for the potsdam dataset are already in data/label/, obtained by transforming the original labels

Feature extraction may take a long time, please be patient and save it

Active learning module development based on modAL (https://github.com/modAL-python/modAL)

Dataset1: FloodNet(https://github.com/BinaLab/FloodNet-Supervised_v1.0)

Dataset2: Potsdam(https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
