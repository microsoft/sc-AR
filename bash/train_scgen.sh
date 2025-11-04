today=$(date '+%Y%m%d')

if [ ! -d log ]; then
  mkdir -p log;
fi

if [ ! -d log/$today ]; then
  mkdir -p log/$today;
fi


# activate environment
eval "$(conda shell.bash hook)"
conda activate scar-env
export WANDB_KEY=''

## Datasets ##
## Species
# data='species'
# list=('mouse' 'pig' 'rabbit' 'rat')

## PBMC
# list=('CD4T' 'B' 'NK' 'CD14+Mono' 'Dendritic' 'CD8T' 'FCGR3A+Mono')
# data='pbmc'

## LPS-Hpoly
data='lps-hpoly'
list=('Endocrine' 'Enterocyte' 'Enterocyte.Progenitor' 'Goblet' 'Stem' 'TA' 'TA.Early' 'Tuft')



## Update these parameters as needed
root='../'
data_path='../data'
out_path='../saved_models/'
# Number of GPUs available
counter=0
num_gpus=1 # 4


## Common Parameters ##
seed_list=(100) # 101 102 103 104)
AR_list=(True False)

if [ $data == 'pbmc' ]; then
    num_epoch=1800
elif [ $data == 'lps-hpoly' ]; then
    num_epoch=2000
elif [ $data == 'species' ]; then
    num_epoch=1000
fi
num_epoch=5

variable_con=False
con_percent=1.0
model_name='scgen'
ood=True

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

## Loop over lists and run train.sh ##
for i in "${!list[@]}"; do
    for seed in "${seed_list[@]}"; do
        for AR in "${AR_list[@]}"; do
            # Assign each iteration to one of the 4 GPUs
            gpu_id=$((counter % num_gpus))
            echo "Running iteration $counter on GPU $gpu_id"
            counter=$((counter + 1))

            (train_data=$(printf '%s\n' "${list[@]:0:$i}" "${list[@]:$(($i + 1))}" | paste -sd ',' -)
            test_data=${list[$i]}
            echo 'data: ' ${data}
            echo 'train_data: ' ${train_data}
            echo 'test_data: ' ${test_data}
            echo 'seed: ' ${seed}
            echo 'AR: ' ${AR}
            echo 'ood: ' ${ood}
            echo 'num_epoch: ' ${num_epoch}
            echo 'variable_con: ' ${variable_con}
            echo 'con_percent: ' ${con_percent}
            echo 'model_name: ' ${model_name}
            echo 'root: ' ${root}
            echo 'gpu_id: ' ${gpu_id}
            echo 'out_path: ' ${out_path}
            echo 'data_path: ' ${data_path}

            CUDA_VISIBLE_DEVICES=${gpu_id} python ../main.py \
                --data ${data} \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --train True \
                --test False \
                --AR ${AR} \
                --ood ${ood} \
                --num_epoch ${num_epoch} \
                --variable_con ${variable_con} \
                --con_percent ${con_percent} \
                --seed ${seed} \
                --data_path ${data_path} \
                --out_path ${out_path} \
                --root ${root} \
                --model_name 'scgen' \
                --model_name ${model_name} >> log/$today/${today}-standard-${data}-test-${list[$i]}-AR${AR}-ood${ood}-seed${seed}-numepoch${num_epoch}-${model_name}-train.txt 2> log/$today/${today}-standard-${data}-test-${list[$i]}-AR${AR}-ood${ood}-seed${seed}-numepoch${num_epoch}-${model_name}-train.err 
                ## add the following input parameters if using a new dataset other than pbmc, lps-hpoly, or species
                ## h5ad_adata_file, this is the full path to the h5ad file including all cell groups
                ## adata_label_cell, this is as string representing the label to be used from the input h5ad file containging cell labels (cell type, species, cell line, etc)
                ## adata_label_per, this is a string representing the label to be used from the input h5ad file marking perturbed cells
                ## adata_label_unper, this is a string representing the label to be used from the input h5ad file marking control cells

            if grep -q "Traceback" log/$today/${today}-standard-${data}-test-${list[$i]}-AR${AR}-ood${ood}-seed${seed}-numepoch${num_epoch}-${model_name}-train.err; then
                echo "Error"
            fi

            if grep -q "main.py: error:" log/$today/${today}-standard-${data}-test-${list[$i]}-AR${AR}-ood${ood}-seed${seed}-numepoch${num_epoch}-${model_name}-train.err; then
                echo "Error"
            fi )&

            # After every 4 processes, wait for them to finish before continuing
            if (( ($counter + 1) % num_gpus == 0 )); then
                wait  # Wait for all background processes to finish
            fi

            done
        done
    done
