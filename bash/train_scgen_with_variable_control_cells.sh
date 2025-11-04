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
export WANDB_KEY=''

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



## Common Parameters ##
seed=(300 301 302 303 304)

num_epoch=500
if [ $data == 'pbmc' ]; then
    num_epoch=1800
elif [ $data == 'lps-hpoly' ]; then
    num_epoch=2000
elif [ $data == 'species' ]; then
    num_epoch=1000
fi


ood=True
AR=(True False)
variable_con=True
con_percent=(0.0 0.2 0.4 0.6 0.8 1.0)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


## Loop over list and run train.sh ##
for i in "${!list[@]}"; do
    for s in "${seed[@]}"; do
        for b in "${AR[@]}"; do
            for p in "${con_percent[@]}"; do

                train_data=$(printf '%s\n' "${list[@]:0:$i}" "${list[@]:$(($i + 1))}" | paste -sd ',' -)
                test_data=${list[$i]}

                echo "data:${data}" 
                echo "train_data:${train_data}"
                echo "test_data:${list[$i]}" 
                echo "AR:${b}"
                echo "OOD:${ood}"
                echo "num_epoch:${num_epoch}" 
                echo "var_con:${variable_con}" 
                echo "con_percent:${p}" 
                echo "seed:${s}"
                echo "root:${root}"
                echo "data_path:${data_path}"
                echo "out_path:${out_path}"
                
                python ../main.py \
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
                    --model_name 'scgen' >> log/$today/${today}-${data}-test${list[$i]}-AR${b}-ood${ood}-varcon${variable_con}-conpercent${p}-seed${s}-train.txt 2> log/$today/${today}-${data}-test${list[$i]}-AR${b}-ood${ood}-varcon${variable_con}-conpercent${p}-seed${s}-train.err

                if grep -q "Traceback" log/$today/${today}-${data}-test${list[$i]}-AR${b}-ood${ood}-varcon${variable_con}-conpercent${p}-seed${s}-train.err; then
                    echo "Error"
                fi

                if grep -q "main.py: error: " log/$today/${today}-${data}-test${list[$i]}-AR${b}-ood${ood}-varcon${variable_con}-conpercent${p}-seed${s}-train.err; then
                    echo "Error"
                fi

                done
            done
        done
    done