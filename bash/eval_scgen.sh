#!/bin/bash

today=$(date '+%Y%m%d')

if [ ! -d log ]; then
  mkdir -p log;
fi

if [ ! -d log/$today ]; then
  mkdir -p log/$today;
fi

# activate environment
eval "$(conda shell.bash hook)"
source activate scar-env


## Datasets ##
## Species
data='species'
list=('mouse' 'pig' 'rabbit' 'rat')

## PBMC
# data='pbmc'
# list=('CD4T' 'B' 'NK' 'CD14+Mono' 'Dendritic' 'CD8T' 'FCGR3A+Mono')

# LPS-Hpoly
# data='lps-hpoly'
# list=('Endocrine' 'Enterocyte' 'Enterocyte.Progenitor' 'Goblet' 'Stem' 'TA' 'TA.Early' 'Tuft')


## Update these parameters as needed
root="../"
data_path="../data"

## Common Parameters ##
seed_list=(100 101 102 103 104)

num_epoch=500
if [ $data == "pbmc" ]; then
    num_epoch=1800
elif [ $data == "lps-hpoly" ]; then
    num_epoch=2000
elif [ $data == "species" ]; then
    num_epoch=1000
fi


model="AR,Naive"
ood=True


for i in "${!list[@]}"; do
    for seed in "${seed_list[@]}"; do

        train_data=$(printf '%s\n' "${list[@]:0:$i}" "${list[@]:$(($i + 1))}" | paste -sd ',' -)
        test_data=${list[$i]}
        echo "data: " $data
        echo "train_data: " $train_data
        echo "test_data: " $test_data
        echo "model: " $model
        echo "ood: " $ood
        echo "num_epoch: " $num_epoch
        echo "seed: " $seed
        echo "model_name: scgen"
        echo "root: " $root

        python ../main.py \
            --root ${root} \
            --data ${data} \
            --train_data ${train_data} \
            --test_data ${test_data} \
            --test True \
            --model ${model} \
            --ood ${ood} \
            --num_epoch ${num_epoch} \
            --model_name "scgen" \
            --seed ${seed} >> log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-seed${seed}-test.txt 2> log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-seed${seed}-test.err


        if grep -q "Traceback" log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-seed${seed}-test.err; then
            echo "Error"
        fi

        if grep -q "main.py: error: " log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-seed${seed}-test.err; then
            echo "Error"
        fi

        done
    done
