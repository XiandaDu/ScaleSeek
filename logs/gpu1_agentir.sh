#!/bin/bash
# 手动补跑 job 7176 里挂掉/没排上的格子，全部丢给已空转的 GPU1。
#
# 修的是什么：sbatch_octal40.sh 的 GPU3 lane 注释写着"先预计算再起 reader，避免显存打架"，
# 但代码是 lane 一开头就 serve4b（0.80×48G=36.8G），之后才做 agentir precompute ——
# 检索模型那 7.5G 根本挤不进去，5 次 precompute 在 0717-09:33 全部 CUDA OOM 秒挂，
# 接着 5 个 agentir 找不到 *_agentir_retrieval.jsonl 全 FAILED。sane() 挡住了假 metrics，
# 所以没脏数据，但格子是空的。
# → 这里把顺序倒过来：先在裸卡上把 6 个数据集的检索缓存全算完（precompute 算完 query
#   向量就 del model + empty_cache，faiss 检索走 CPU），再起 4B reader。
#
# 顺带把 hotpotqa 缺的 direct / dci 两格接上（同一个 4B 服务，不用另起）。
# hotpotqa_grepseek 要 9B，等 GPU0 全量跑完（~07:00）再补，不在本脚本内。
set -u
# —— 硬规则：绝不在跳板机(arcade*)跑重活 ——
case "$(hostname)" in
  octal40*) : ;;
  *) echo "REFUSING: 当前在 $(hostname)，本脚本只能在 octal40 跑"; exit 1;;
esac
PY=/u/mofengra/miniconda3/envs/scaleseek/bin/python
cd /data/rech/mofengra/ScaleSeek || exit 1
# L40S(sm_89) 两连坑，不设这两个 vLLM 起不来且是静默失败（见 node_cuda_home 记忆）
export CUDA_HOME=/u/mofengra/miniconda3/envs/scaleseek/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=12
export HF_HOME=/data/rech/mofengra/data/hf_cache HF_HUB_CACHE=/data/rech/mofengra/data/hf_cache/hub HF_HUB_OFFLINE=1
export DATASETS=/data/rech/mofengra/datasets LLM_TOKENIZER=Qwen/Qwen3-4B
export E5_INDEX_DIR=/data/rech/mofengra/data/e5_index E5_DEVICE=cuda
export BM25_INDEX_DIR=/data/rech/mofengra/data/bm25_index
AIDX=/data/rech/mofengra/data/agentir_index_v2
CORPUS=/data/rech/mofengra/data/wiki_18_corpus/wiki_corpus.jsonl
TDB=/data/rech/mofengra/data/corpus_title_index.db
PORT=8066
DS="nq triviaqa 2wikimultihopqa musique bamboogle hotpotqa"
tf(){ case "$1" in hotpotqa|2wikimultihopqa) echo "--title-index-db $TDB";; esac; }
sane(){ local out=$1 t e
  t=$(wc -l < results/$out.jsonl 2>/dev/null | tr -d ' '); [ "${t:-0}" -gt 0 ] || { echo "[g1] $out 空文件"; return 1; }
  e=$(grep -c '"finish_reason": "api_error"' results/$out.jsonl 2>/dev/null | head -1 | tr -d ' '); e=${e:-0}
  [ $(( e * 100 / t )) -lt 50 ] || { echo "[g1] $out api_error $e/$t 过半 —— 无效，不出 metrics"; return 1; }; }

# 释放 GPU1：search_r1 lane 11/11 格已于 0717-20:13 全部完成，3B 服务纯占着 37G 空转。
# 只 kill 8062，不碰 8061(GPU0 全量) / 8064(GPU3 dci) / 8065(GPU2 grepseek)。
echo "[g1] 释放 GPU1：kill 8062 已完工的 3B search_r1 ..."
pkill -f "[v]llm.entrypoints.*--port 8062" 2>/dev/null; sleep 10

# ---- 阶段 1：裸卡上算 agentir 检索缓存（此时 GPU1 上没有任何 vLLM，不会 OOM）----
for ds in $DS; do
  c=results/${ds}_agentir_retrieval.jsonl
  [ -f $c ] && { echo "[g1] $ds 检索缓存已存在，跳过"; continue; }
  echo "[g1] precompute $ds start @ $(date +%m%d-%H:%M)"
  CUDA_VISIBLE_DEVICES=1 $PY scripts/precompute_agentir_retrieval.py --dataset $ds -n 1500 \
    --index-root $AIDX --top-k 5 --device cuda --out $c \
    && echo "[g1] precompute $ds done @ $(date +%m%d-%H:%M)" \
    || { echo "[g1] precompute $ds FAILED"; rm -f $c; }
done

# ---- 阶段 2：缓存都算完了，再起 4B reader 吃下整张卡 ----
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name agent --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 --gpu-memory-utilization 0.80 \
  --max-model-len 32768 > logs/g1_4b.log 2>&1 &
ok=0; for t in $(seq 1 180); do curl -sf -m 5 localhost:$PORT/v1/models >/dev/null 2>&1 && { ok=1;break;}; sleep 10; done
[ $ok = 1 ] || { echo "[g1] 4B 未起来（查 logs/g1_4b.log 有没有 nvcc/headers 报错）ABORT"; exit 1; }
echo "[g1] 4B up @ $(date +%m%d-%H:%M)"

cell(){ local out=$1; shift
  [ -f results/$out.metrics.json ] && { echo "[g1] $out 已完成，跳过"; return; }
  echo "[g1] $out start @ $(date +%m%d-%H:%M)"
  "$@" --resume --output results/$out.jsonl \
    && sane $out \
    && $PY scripts/compute_metrics.py --results results/$out.jsonl $TFARG --out results/$out.metrics.json \
    && echo "[g1] $out done @ $(date +%m%d-%H:%M)" || echo "[g1] $out FAILED"; }

# 6 格 agentir（就是 OOM 掉的那批 + hotpotqa 补齐主表）
for ds in $DS; do
  TFARG=$(tf $ds)
  [ -f results/${ds}_agentir_retrieval.jsonl ] || { echo "[g1] ${ds}_agentir 无缓存，跳过"; continue; }
  cell ${ds}_agentir env CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset $ds --agent agentir_rag \
    -n 1500 --concurrency 16 --port $PORT --agentir-cache results/${ds}_agentir_retrieval.jsonl \
    --max-tokens 2048
done

# hotpotqa 主表缺的另两格（grepseek 那格要 9B，等 GPU0 空出来再补）
TFARG=$(tf hotpotqa)
cell hotpotqa_direct env CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset hotpotqa --agent direct \
  -n 1500 --concurrency 16 --port $PORT --retrieval-backend bm25 --max-tokens 2048
cell hotpotqa_dci env CUDA_VISIBLE_DEVICES=1 $PY -m eval.run_eval --dataset hotpotqa --agent dci \
  -n 1500 --concurrency 4 --port $PORT --corpus-path $CORPUS --max-tokens 2048

pkill -f "[v]llm.entrypoints.*--port $PORT"
echo "G1_ALL_DONE @ $(date +%m%d-%H:%M)"
