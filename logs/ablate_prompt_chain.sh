#!/bin/bash
# scaleseek prompt 消融（任务 #11 + #12 的生成 max_tokens 探针）
# A = scaleseek_prompt_noparams（零数字，只解释参数含义）
# B = scaleseek_prompt（现状，含参考数值）
# 域：popqa_full(单跳) / hotpotqa(多跳) / browsecomp_plus(BCP)，各 n=500（IRCoT 先例）
# 额外：B 变体 popqa 生成 max_tokens 4096 探针（默认 2048 对照）
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
WIKI_IDX=/data/rech/mofengra/data/bm25_index
BCP_IDX=/data/rech/mofengra/data/bcp_bm25_index
QRELS="--bcp-qrels $DATASETS/browsecomp_plus/qrels.json --bcp-doclen $DATASETS/browsecomp_plus/doclen.json"

run_cell() {  # $1=variant(A|B) $2=dataset $3=outname $4...=extra
  local var=$1 ds=$2 out=$3; shift 3
  if [ "$var" = A ]; then export SCALESEEK_PROMPT=scaleseek_prompt_noparams
  else unset SCALESEEK_PROMPT; fi
  echo "[ablate-prompt] $out start @ $(date +%m%d-%H:%M)"
  $PY -m eval.run_eval --dataset $ds --agent scaleseek -n 500 --concurrency 8 \
    --output results/$out.jsonl "$@" \
    || { echo "[ablate-prompt] $out FAILED"; return 1; }
  local extra_metrics=""
  [ "$ds" = browsecomp_plus ] && extra_metrics="$QRELS"
  $PY scripts/compute_metrics.py --results results/$out.jsonl $extra_metrics \
    --out results/$out.metrics.json
  echo "[ablate-prompt] $out done @ $(date +%m%d-%H:%M)"
}

export BM25_INDEX_DIR=$WIKI_IDX
run_cell A popqa_full ablate_ss_A_popqa_500
run_cell B popqa_full ablate_ss_B_popqa_500
run_cell A hotpotqa   ablate_ss_A_hotpot_500
run_cell B hotpotqa   ablate_ss_B_hotpot_500
run_cell B popqa_full ablate_ss_B4096_popqa_500 --max-tokens 4096

export BM25_INDEX_DIR=$BCP_IDX
run_cell A browsecomp_plus ablate_ss_A_bcp_500 --max-tokens 4096
run_cell B browsecomp_plus ablate_ss_B_bcp_500 --max-tokens 4096
echo "ABLATE_PROMPT_ALL_DONE @ $(date +%m%d-%H:%M)"
