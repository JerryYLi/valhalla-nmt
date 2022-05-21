# usage: bash test.sh [-s src] [-t tgt] [-m mode] [-a arch] [-k ckps] [-y subsets] [-n ckp_name] [-g gt_image]
src=en
tgt=de
mode=valhalla
arch=vldtransformer_tiny
subsets="test test1 test2"
ckps=1
ckp_name=checkpoint_best.pt
while getopts ":s:t:m:a:k:y:n:g" opt; do
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
    y) subsets="$OPTARG"
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

task=multi30k/${src}-${tgt}
data_dir=data-bin/${task}
for subset in ${subsets[@]}; do
    save_name=${subset}_${key}${gt}
    for a in ${arch[@]}; do
        ckp_root=checkpoints/${task}/${mode}/${a}
        for ckp in $(find ${ckp_root} -name ${ckp_name}); do
            ckp_dir=$(dirname ${ckp})
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
            if [[ "${subset}" == "test2" ]]; then
                vis_data=mscoco
                vis_args="{\"resize\": 128, \"crop\": 128}"
            elif [[ "${subset}" == "test1" ]]; then
                vis_data=flickr30k
                vis_args="{\"test_year\": 2017, \"resize\": 128, \"crop\": 128}"
            else
                vis_data=flickr30k
                vis_args="{\"test_year\": 2016, \"resize\": 128, \"crop\": 128}"
            fi
            # do not load visual dataset unless using multimodal inference
            if [ -z ${gt} ]; then
                vis_data=none
            fi
            echo "Evaluating model ${ckp_eval} on ${subset} split. Image data: ${vis_data}"
            output=${ckp_dir}/${save_name}.pred
            if [ ! -f ${output} ]; then
                python generate.py ${data_dir} --gen-subset ${subset} \
                --task vislang_translation --source-lang ${src} --target-lang ${tgt} \
                --model-overrides "{${gt_opt}}" \
                --vis-data ${vis_data} --vis-data-args "${vis_args}" \
                --path ${ckp_eval} --beam 5 --remove-bpe --lenpen 1 \
                --batch-size 128 > ${output}
            fi
            tail -n 2 ${output}
        done
    done
done