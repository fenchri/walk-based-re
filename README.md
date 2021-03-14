
# Walk-based Relation Extraction
Source code for the ACL 2018 paper "[A Walk-based model on Entity Graphs for Relation Extraction](https://www.aclweb.org/anthology/P18-2014/)".


### Requirements & Environment
```
pip3 install -r requirements.txt
```
The original model of the paper was implement in [Chainer](https://chainer.org/). This is the same version of the model in [PyTorch](https://pytorch.org/).
Slight differences might occur due to the above change.

#### Reproducability
Results are reproducible with the a fixed seed.
Experimentation showed performance increased significantly when classifying only 1 direction (in comparison with the paper, where both directions are classified). 
If you want to reproduce the results of the paper you need to use the `--direction l2r+r2l` argument.
Otherwise, we recommend to use `--direction r2l` or `--direction l2r` in the input or in the config files.




## Data & Pre-processing
Download (using the appropriate license) the [ACE 2005 dataset](https://catalog.ldc.upenn.edu/LDC2006T06).
Clone the [LSTM-ER](https://github.com/tticoin/LSTM-ER) repository in order to pre-process the data.
```
$ cd data_processing/
$ git clone https://github.com/tticoin/LSTM-ER.git
$ cd LSTM-ER/data/ && mkdir common/ && cd common/
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip && unzip stanford-corenlp-full-2015-04-20.zip
$ wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip && unzip stanford-postagger-2015-04-20.zip
$ cd ..
```

Place the ACE 2005 dataset original data `English/` folder into `data_processing/LSTM-ER/data/ace2005/` and run,
```
$ zsh preprocess_ace05.zsh 
$ sh process_ace05.sh
```

Download the pre-trained word embeddings:
```
$ mkdir embeds/ && cd embeds
$ wget http://tti-coin.jp/data/wikipedia200.bin
$ python3 ../data_processing/bin2txt.py wikipedia200.bin   # convert to .txt
$ cd ..
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
usage: walk_re.py [-h] --config CONFIG [--train] [--test] --gpu GPU
                  [--walks WALKS] [--att {True,False}] [--example]
                  [--direction {l2r,r2l,l2r+r2l}] [--folder FOLDER]
                  [--embeds EMBEDS] [--train_data TRAIN_DATA]
                  [--test_data TEST_DATA] [--epoch EPOCH] [--early_stop]
                  [--preds PREDS]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Yaml parameter file
  --train               Training mode - model is saved
  --test                Testing mode - needs a model to load
  --gpu GPU             GPU number, use -1 for CPU
  --walks WALKS         Number of walk iterations
  --att {True,False}    Use attention or not
  --example             Print the sentences and info in the 1st batch, then
                        exit (useful for debugging)
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
  --early_stop          Use early stopping
  --preds PREDS         Folder name for predictions
```


For model tuning, download and install the [RoBO toolkit](https://github.com/automl/RoBO), then run:
```
$ cd src/bin
$ ./tune_ace05.sh
```


### Citation
Please cite the following paper when using this code:

```
@inproceedings{christopoulou2018walk,  
title={A Walk-based Model on Entity Graphs for Relation Extraction},  
author={Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia},  
booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},  
year={2018},  
publisher={Association for Computational Linguistics},  
pages={81--88},  
}
```

