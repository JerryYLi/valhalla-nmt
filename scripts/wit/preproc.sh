# usage: bash preproc.sh [-s src] [-t tgt]
src=en
tgt=de
bpe_tokens=10000
while getopts ":s:t:b:" opt; do
  case $opt in
    s) src="$OPTARG"
    ;;
    t) tgt="$OPTARG"
    ;;
    b) bpe_tokens="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

task=${src}_${tgt}
text=data/wit/mmt/${task}/bpe${bpe_tokens}
python fairseq/fairseq_cli/preprocess.py --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${text}/train --validpref ${text}/valid --testpref ${text}/test \
    --joined-dictionary --workers 16 --destdir data-bin/wit/${task}/bpe${bpe_tokens}