#!/bin/bash
# BCP 冒烟链：50 题 × {direct, bm25_rag(1.2/0.75), bm25_rag(25/1.0), scaleseek,
# grepseek}，顺序执行（conc 8，给 DR-DCI 830 留 vLLM 余量）。
# 指标：EM/F1 + Gold/Qrel R@W + Coverage/Localization（BCP qrels 全支持）。
case "$(hostname)" in
  octal30*|octal35*) : ;;
  *) echo "REFUSING non-GPU host: $(hostname)"; exit 1 ;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets
export BM25_INDEX_DIR=/data/rech/mofengra/data/bcp_bm25_index
BCP_CORPUS=$DATASETS/browsecomp_plus/corpus.jsonl
QRELS="--bcp-qrels $DATASETS/browsecomp_plus/qrels.json --bcp-doclen $DATASETS/browsecomp_plus/doclen.json"

run_one() {  # $1=agent $2=outfile $3...=extra flags
  local agent=$1 out=$2; shift 2
  echo "[bcp-chain] $agent -> $out start @ $(date +%m%d-%H:%M)"
  $PY -m eval.run_eval --dataset browsecomp_plus --agent $agent -n 50 \
    --concurrency 8 --output results/$out.jsonl "$@" \
    || { echo "[bcp-chain] $agent FAILED"; return 1; }
  $PY scripts/compute_metrics.py --results results/$out.jsonl $QRELS \
    --out results/$out.metrics.json
  echo "[bcp-chain] $agent done @ $(date +%m%d-%H:%M)"
}

run_one direct     bcp_direct_smoke50
run_one bm25_rag   bcp_bm25_smoke50            --bm25-k1 1.2 --bm25-b 0.75
run_one bm25_rag   bcp_bm25_k1-25_b-1.0_smoke50 --bm25-k1 25 --bm25-b 1.0
run_one scaleseek  bcp_scaleseek_smoke50
run_one grepseek   bcp_grepseek_smoke50        --corpus-path $BCP_CORPUS
echo "BCP_CHAIN_ALL_DONE @ $(date +%m%d-%H:%M)"
