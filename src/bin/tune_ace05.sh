#!/bin/bash

cd ../

declare -a directions=("r2l")
gpu=0
w=2
l=$((2 ** ${w}))


for w in 0 1 2 3;
do
    l=$((2 ** ${w}))

    for dir in ${directions[@]};
    do
        python3 hyperparam_tuning.py --config ../configs/ace2005_params_l${l}.yaml  \
                                     --train_data ../../RELATION_DATA/ACE-2005/train.data \
                                     --test_data ../../RELATION_DATA/ACE-2005/dev.data \
                                     --embeds ../../TRAINED-EMBEDS/wikipedia200_ace05.txt \
                                     --walks ${w} \
                                     --train \
                                     --att vector \
                                     --folder ../saved_models/ace05_dev_l${l} \
                                     --direction ${dir} \
                                     --early_stop \
                                     --gpu ${gpu}
    done
done
