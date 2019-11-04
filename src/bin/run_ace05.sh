#!/usr/bin/env bash

cd ../

declare -a directions=("r2l")
gpu=0


# ======================= TRAINING ======================= #

# + context + walks (test the model model, all things included)
for w in 0 1 2 3;
do
    l=$((2 ** ${w}))


    for dir in ${directions[@]};
    do
        python3 walk_re.py --config ../configs/ace2005_params_l${l}.yaml  \
                           --train_data ../../RELATION_DATA/ACE-2005/train.data \
                           --test_data ../../RELATION_DATA/ACE-2005/dev.data \
                           --embeds ../../TRAINED-EMBEDS/wikipedia200_ace05.txt \
                           --walks ${w} \
                           --test \
                           --att vector \
                           --folder ../saved_models/ace05_dev_l${l} \
                           --direction ${dir} \
                           --early_stop \
                           --gpu ${gpu}
    done
done


# ======================= TESTING ======================= #


for w in 0 1 2 3;
do
    l=$((2 ** ${w}))
    for dir in ${directions[@]};
    do
        x="beta????"
        ep=`python3 analysis/find_best_epoch.py ../saved_models/ace05_dev_l${l}/${x}-walks${l}-att_${att}-dir_${dir}/info_train.log`
        if (( "${ep}" != "-1" )); then
            echo ${ep}

            python3 walk_re.py --config ../configs/ace2005_params_l${l}.yaml  \
                               --train_data ../../RELATION_DATA/ACE-2005/train+dev.data \
                               --test_data ../../RELATION_DATA/ACE-2005/test.data \
                               --embeds ../../TRAINED-EMBEDS/wikipedia200_ace05.txt \
                               --walks ${w} \
                               --epoch ${ep} \
                               --att vector \
                               --train \
                               --folder ../saved_models/ace05_test_l${l} \
                               --preds test \
                               --direction ${dir} \
                               --gpu ${gpu}
        fi
    done
done
