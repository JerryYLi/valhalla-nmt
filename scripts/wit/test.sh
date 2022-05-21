# usage: bash test.sh [-s src] [-t tgt] [-m mode] [-a arch] [-k ckps] [-b tokens] [-n ckp_name] [-g gt_image]
src=en
tgt=de
mode=valhalla
arch=vldtransformer
tokens=10000
ckps=1
ckp_name=checkpoint_best.pt
vis_data=wit
while getopts ":s:t:m:a:k:b:n:g" opt; do
  case $opt in
    s) src="$OPTARG"
    ;;
    t) tgt="$OPTARG"
    ;;
    m) mode="$OPTARG"
    ;;
    a) arch="$OPTARG"
    ;;
    k) ckps="$OPTARG"
    ;;
    b) tokens="$OPTARG"
    ;;
    n) ckp_name="$OPTARG"
    ;;
    g) 
        gt="_mmt"
        gt_opt="\"mmt_inference\": True"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "${ckps}" -gt 1 ]; then
    key="avg${ckps}"
else
    key="best"
fi

task=wit/${src}_${tgt}
save_name=test_${key}${gt}.pred
data_dir=data-bin/${task}/bpe${tokens}
for a in ${arch[@]}; do
    ckp_root=checkpoints/${task}/${mode}/${a}
    for ckp in $(find ${ckp_root} -name ${ckp_name}); do
        # compute average checkpoint, if enabled
        if [ "${ckps}" -gt 1 ]; then
            ckp_eval="${ckp_dir}/checkpoint_avg${ckps}.pt"
            if [ ! -f ${ckp_eval} ]; then
                echo "Generating average model ${ckp_eval}"
                bash scripts/tools/average_ckp.sh ${ckp_dir} epoch ${ckps}
            fi
        else
            ckp_eval=${ckp}
        fi
        # evaluate model
        vis_args="{\"resize\": 128, \"crop\": 128, \"normalize\": false}"
        if [ -z ${gt} ]; then
            vis_data=none
        fi
        echo "Evaluating model ${ckp} on test split. Image data: ${vis_data}"
        output=$(dirname ${ckp})/${save_name}
        if [ ! -f ${output} ]; then
            python generate.py ${data_dir} \
            --task vislang_translation --source-lang ${src} --target-lang ${tgt} \
            --target-lang ${tgt} \
            --model-overrides "{${gt_opt}}" \
            --vis-data ${vis_data} --vis-data-args "${vis_args}" \
            --path ${ckp} --beam 5 --remove-bpe --lenpen 2 \
            --batch-size 128 > ${output}
        fi
        tail -n 2 ${output}
    done
done
