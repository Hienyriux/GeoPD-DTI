# GeoPD-DTI
Graph Neural Networks with Geometric Edge Fusion and Point Downsampling for Drug-Target Interaction Prediction (BIBM 2023, Short)

## Requirements
* python == 3.8.13
* pytorch == 1.11.0
* scikit-learn == 1.1.1
* torch-geometric == 2.0.4

## Usage

### Data Preprocessing
Please download the dataset file from the url, and unzip it. If you successfully downloaded and unzipped the file, you can skip the following data preprocessing steps and directly run the commands that will be introduced later to train or test the model. If you want to generate those files from scratch, you should first clone the repository [DeepDTA](https://github.com/hkmztrk/DeepDTA/tree/master) and copy the `data` directory to the `dataset` directory of this repository, and rename the copied `data` directory as `raw_data`. Then, run the code in the following files in sequence:

* `utils.py`
* `utils_mol_graph.py`
* `utils_fp.py`
* `utils_esm.py`
* `utils_esmfold.py`
* `utils_spatial.py`

When you run a file, you should sequentially uncomment one code line at the end of the file while keeping other lines commented and then run the code once. If you want to run `utils_esm.py` or `utils_esmfold.py`, make sure ESMfold related environments are correctly installed, please see the [ESM](https://github.com/facebookresearch/esm) repository for details. You should place all ESM model weight files in the `checkpoints` directory (please create this directory) in the root directory of this repository.

### Train, Test, and Cross Validation
You can train the model using the following command:
`python geopd_dti.py --dataset_name davis --mode train --fold 0 --hidden_dim 128 --num_epoch 1000 --batch_size 256 --lr 0.001 --seed 0 --device cuda`

Here's the explanation of the arguments:
* `--dataset_name`: the name of the dataset you want to use, it can be `davis` or `kiba`
* `--mode`: running mode, which can be `train`, `test`, or `train_valid`, for training, test, or five-fold cross validation, respectively
* `--fold`: the cross validation fold, can be 0, 1, 2, 3, 4, only activated in `train_valid` mode
* `--hidden_dim`: the hidden dimensionality of the model
* `--num_epoch`: the number of epochs you want to run for training or cross validation. In `tes`t mode, it represents the number of epochs for which the loaded model was trained
* `--batch_size`: the size of the batch
* `--lr`: the (base) learning rate. Note that we use cyclic learning rate scheduling during training
* `--seed`: the seed to control the generation of random numbers during training
* `--device`: the device where the model will be run, e.g., `cuda`, `cuda:1`, `cpu`, etc. We recommend you to run the model on GPU to avoid unnecessary trouble.

You can test the model using the following command:
`python geopd_dti.py --dataset_name davis --mode test --fold 0 --hidden_dim 128 --num_epoch 300 --batch_size 256 --lr 0.001 --seed 0 --device cuda`

You can do cross validation using the following command:
`python geopd_dti.py --dataset_name davis --mode train_valid --fold 0 --hidden_dim 128 --num_epoch 1000 --batch_size 256 --lr 0.001 --seed 0 --device cuda`

## License
MIT
