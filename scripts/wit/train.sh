# usage: bash train.sh [-s src] [-t tgt] [-a arch] [-b tokens] [-q vis_tokens] [-n batch_size] [-l learning_rate]
#                      [-v vis_arch] [-w vis_ckp] [-x vis_cfg] [-k vis_dim] [-c crop] [-h vis_height]
#                      [-u halluc_ckp] [-z halluc_offline] [-r gumbel_hard]
#                      [-e consistency_weight] [-g halluc_weight] [-f suffix] [-o args] [-j num_workers]
src=en
tgt=de
arch=vldtransformer
halluc=dalle
vis_arch=vqgan
vis_ckp=path/to/vis_ckp.pth
vis_cfg=path/to/vis_cfg.yaml
vis_dim=128
crop=128
vis_height=4
tokens=10000
vis_tokens=1024
halluc_ckp=path/to/halluc_ckp.pth
consistency_weight=0.5
halluc_weight=0.5
halluc_online=true
gumbel_hard=false
bs=1024
lr=0.0005
opts=base
num_workers=16
while getopts ":s:t:a:b:q:n:l:v:w:x:k:c:h:u:zre:g:f:o:j:" opt; do
  case $opt in
    s) src="$OPTARG"
    ;;
    t) tgt="$OPTARG"
    ;;
    a) arch="$OPTARG"
    ;;
    b) tokens="$OPTARG"
    ;;
    q) vis_tokens="$OPTARG"
    ;;
    n)
      bs="$OPTARG"
      opts="${opts}_bs${bs}"
      ;;
    l)
      lr="$OPTARG"
      opts="${opts}_lr${lr}"
      ;;
    v) 
      vis_arch="$OPTARG"
      ;;
    w) 
      vis_ckp="$OPTARG"
      ;;
    x) 
      vis_cfg="$OPTARG"
      ;;
    k) 
      vis_dim="$OPTARG"
      ;;
    c) 
      crop="$OPTARG"
      opts="${opts}_vc${crop}"
      ;;
    h) 
      vis_height="$OPTARG"
      opts="${opts}_vh${vis_height}"
      ;;
    u) 
      halluc_ckp="$OPTARG"
      ;;
    z) 
      halluc_online=false
      opts="${opts}_offline"
      ;;
    r) 
      gumbel_hard=true
      opts="${opts}_hard"
      ;;
    e) 
      consistency_weight="$OPTARG"
      opts="${opts}_dst${consistency_weight}"
      ;;
    g) 
      halluc_weight="$OPTARG"
      opts="${opts}_hal${halluc_weight}"
      ;;
    f) suffix="$OPTARG"
    ;;
    o) args="$OPTARG"
    ;;
    j) num_workers="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

task=wit/${src}_${tgt}
data_dir=data-bin/${task}/bpe${tokens}
vis_data=wit
vis_dir=data/${vis_data}

# Create checkpoint and logging dirs
ckp=${task}/valhalla/${arch}/${vis_arch}/${opts}${suffix}
ckp_dir=checkpoints/${ckp}
tb_dir=tb-logs/${ckp}
echo Checkpoint: ${ckp_dir}
mkdir -p ${ckp_dir} ${tb_dir}

python train.py ${data_dir} --task vislang_halluc_translation \
  -s ${src} -t ${tgt} --share-all-embeddings \
  --arch ${arch} --vis-encoder-embed-dim ${vis_dim} --vis-encoder-hallucinate ${halluc} \
  --vis-data ${vis_data} --vis-data-args "{\"root\": \"${vis_dir}\", \"resize\": ${crop}, \"crop\": ${crop}}" \
  --vis-encoder-arch ${vis_arch} --vis-encoder-model-path ${vis_ckp} --vis-encoder-config-path ${vis_cfg} \
  --vis-encoder-grid-h ${vis_height} --vis-encoder-tokens ${vis_tokens} \
  --halluc-model-path ${halluc_ckp} --halluc-args "{\"online\": ${halluc_online}, \"gumbel_hard\": ${gumbel_hard}}" \
  --optimizer adam --adam-betas '(0.9, 0.98)' --patience 20 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --lr ${lr} --stop-min-lr 1e-09 --dropout 0.3 \
  --criterion label_smoothed_cross_entropy_halluc \
  --label-smoothing 0.1 --consistency-loss kld \
  --consistency-weight ${consistency_weight} --halluc-weight ${halluc_weight} \
  --max-tokens ${bs} --max-update 50000 --update-freq 1 \
  --no-progress-bar --log-interval 1000 --log-format simple \
  --save-interval-updates 1000 --keep-interval-updates 5 --keep-last-epochs 5 \
  --save-dir ${ckp_dir} --tensorboard-logdir ${tb_dir} \
  --num-workers ${num_workers} ${args}