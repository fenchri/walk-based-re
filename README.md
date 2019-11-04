
# Walk-based RE
Source code for the paper "[A Walk-based model on Entity Graphs for Relation Extraction](https://www.aclweb.org/anthology/P18-2014/)" in ACL 2018.


### Requirements
- python 3.5 +
- PyTorch 1.1.0
- tqdm
- matplotlib
- pyyaml
- recordtype
- yamlordereddictloader
- gensim (only for converting the .bin embeddings to .txt)
```
pip3 install -r requirements.txt
```

### Environment
The original model of the paper was implement in [Chainer](https://chainer.org/). This is the same version of the model on [PyTorch](https://pytorch.org/).
Results might not be exactly the same as in the paper due to this change.
Further experimentation showed the performance increased significantly when classifying only 1 direction (in comparison with the paper, where we classify both directions). 
You can select this option from the "--direction" argument in the config files.




## Data & Pre-processing
Download (using the appropriate license) the [ACE 2005 dataset](https://catalog.ldc.upenn.edu/LDC2006T06).
Use the pre-processing code from the [LSTM-ER](https://github.com/tticoin/LSTM-ER) repo and then run:
```
$ cd data_processing
$ python3 process.py --input_folder folder_with_ace05_train_data --output_file ../data/ACE-2005/train.data --domain gen --processed 
```

Download the pre-trained word embeddings:
```
$ mkdir embeds/
$ cd embeds
$ wget http://tti-coin.jp/data/wikipedia200.bin
$ cd ../data_processing
$ python3 bin2txt.py ../embeds/wikipedia200.bin   # convert to .txt
```

## Usage
Run the main script for training or testing as follows:
```
$ cd src/
$ python3 walk_re.py --config ../configs/ace2005_params_l4.yaml --train --gpu 0
$ python3 walk_re.py --config ../configs/ace2005_params_l4.yaml --test --gpu 0
```
Alternatively one can use the bash script:
```
$ cd src/bin
$ ./run_ace05.sh   # run multiple models train + testing
```
A portion of the model parameters can be given directly from the command line as follows:
```
usage: walk_re.py [-h] --config CONFIG [--train] [--test] [--gpu GPU]
                  [--walks WALKS] [--att {vector}] [--example]
                  [--direction {l2r,r2l,l2r+r2l}] [--folder FOLDER]
                  [--embeds EMBEDS] [--train_data TRAIN_DATA]
                  [--test_data TEST_DATA] [--epoch EPOCH] [--early_stop]
                  [--preds PREDS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Yaml parameter file
  --train               Training mode - model is saved
  --test                Testing mode - needs a model to load
  --gpu GPU             GPU number. Use -1 for CPU.
  --walks WALKS         Number of walk iterations
  --att {vector}        attention type
  --example             Print the sentences and info in the 1st batch, then
                        exit.
  --direction {l2r,r2l,l2r+r2l}
                        Direction of arguments to classify
  --folder FOLDER       Destination folder to save model, predictions and
                        errors
  --embeds EMBEDS       Pre-trained word embeds file
  --train_data TRAIN_DATA
                        Training data file
  --test_data TEST_DATA
                        Test data dile
  --epoch EPOCH         Stopping epoch
  --early_stop
  --preds PREDS         Forder name for predictions
```


For model tuning, download and install the [RoBO toolkit](https://github.com/automl/RoBO), then run:
```
$ cd src/bin
$ ./tune_ace05.sh
```


### Citation
Please cite the following paper when using this code:

> @inproceedings{christopoulou2018walk,  
title={A Walk-based Model on Entity Graphs for Relation Extraction},  
author={Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia},  
booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},  
year={2018},  
publisher={Association for Computational Linguistics},  
pages={81--88},  
}

