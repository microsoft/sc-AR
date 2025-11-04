today=$(date '+%d%m%Y')

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

## LPS-Hpoly
# data='lps-hpoly'
# list=('Endocrine' 'Enterocyte' 'Enterocyte.Progenitor' 'Goblet' 'Stem' 'TA' 'TA.Early' 'Tuft')


num_epoch=0
if [ $data == 'pbmc' ]; then
    num_epoch=1800
elif [ $data == 'lps-hpoly' ]; then
    num_epoch=2000
elif [ $data == 'species' ]; then
    num_epoch=1000
fi


## Common Parameters ##
seed=(100 101 102 103 104) # list of seeds
model='AR,Naive'
ood=True
today=$(date '+%d%m%Y')
variable_con=False
con_percent=1.0
root='../'
data_path='../data'
out_path='../saved_models/'


## Loop over list and run train_variable_con_percent.sh ##
for i in "${!list[@]}"; do
    test_data="${list[$i]}"
    train_data=$(printf '%s\n' "${list[@]:0:$i}" "${list[@]:$((i + 1))}" | paste -sd ',' -)

    IFS=',' read -ra train_array <<< "$train_data"
    for in_dist_group in "${train_array[@]}"; do
        for s in "${seed[@]}"; do

            echo 'data: ' $data
            echo 'train_data: ' $train_data
            echo 'test_data: ' $test_data
            echo 'in_dist_group: ' $in_dist_group
            echo 'model: ' $model
            echo 'ood: ' $ood
            echo 'num_epoch: ' $num_epoch
            echo 'variable_con: ' $variable_con
            echo 'con_percent: ' $con_percent
            echo 'seed: ' $s
            echo 'root: ' $root
            echo 'data_path: ' $data_path
            echo 'out_path: ' $out_path
            echo 'model_name: scgen'

            python ../main.py \
                --data ${data} \
                --train_data ${train_data} \
                --test_data ${test_data} \
                --in_dist_group "${in_dist_group}" \
                --predict True \
                --model ${model} \
                --ood ${ood} \
                --num_epoch ${num_epoch} \
                --variable_con ${variable_con} \
                --con_percent ${con_percent} \
                --model_name 'scgen' \
                --root ${root} \
                --data_path ${data_path} \
                --out_path ${out_path} \
                --seed ${s} >> ./log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-varcon${variable_con}-conper${con_percent}-seed${s}-indist${in_dist_group}-predict.txt 2> ./log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-varcon${variable_con}-conper${con_percent}-seed${s}-indist${in_dist_group}-predict.err

            if grep -q "Traceback" ./log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-varcon${variable_con}-conper${con_percent}-seed${s}-indist${in_dist_group}-predict.err; then
                echo "Error"
            fi

            if grep -q "main.py: error: " ./log/$today/${today}-standard-${data}-test-${list[$i]}-${model}-OOD${ood}-epoch${num_epoch}-varcon${variable_con}-conper${con_percent}-seed${s}-indist${in_dist_group}-predict.err; then
                echo "Error"
            fi

            echo "========================================================================================="
            echo ""
        done
    done
done
    