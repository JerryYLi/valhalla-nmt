# usage: bash average_ckp.sh [checkpoint path] [mode: update|epoch] [avg_n]
CKP=$1
MODE=${2-update}
AVG=${3-10}
python fairseq/scripts/average_checkpoints.py \
    --inputs $CKP \
    --output $CKP/checkpoint_avg$AVG.pt \
    --num-$MODE-checkpoints $AVG