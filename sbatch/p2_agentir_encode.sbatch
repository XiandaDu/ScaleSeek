#!/usr/bin/env bash
#SBATCH -J p2_agentir_encode
#SBATCH -p rali
# NOTE: was `-w octal30`. Pinning a single node meant the job could not run at
# all once that node went down (2026-08-09). octal31 is the identical 4x A5000
# box; let the scheduler choose among equivalent nodes instead.
#SBATCH --constraint=rtx_a5000
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 7-00:00:00
#SBATCH --requeue
#SBATCH -o /data/rech/mofengra/ScaleSeek/logs/%x-%j.out
# AgentIR prerequisite: encode the full Wiki-18 corpus with the frozen
# AgentIR-4B checkpoint via the official Tevatron encode driver, producing the
# corpus.*.pkl shards that the official FaissSearcher expects.
# The obsolete self-built agentir_index_v2 is intentionally NOT reused.
source /data/rech/mofengra/ScaleSeek/sbatch/common.sh
trap cleanup_tmpdir EXIT
OUT=$DATA/agentir_official_index
SNAP=$HF_HUB_CACHE/models--Tevatron--AgentIR-4B/snapshots/e31abb637caa227c4b7d04176a24ecff1bcb10f4
mkdir -p "$OUT"

echo "== 1. corpus -> tevatron jsonl (docid/text) =="
TEV=$DATA/agentir_official_index/corpus_tevatron.jsonl
if [ ! -f "$TEV.done" ]; then
  "$PY" - "$CORPUS_FILE" "$TEV" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        fout.write(json.dumps({"docid": str(row["id"]), "title": "",
                               "text": row.get("contents", "")},
                              ensure_ascii=False) + "\n")
print("converted", dst)
PYEOF
  touch "$TEV.done"
fi

echo "== 2. tevatron encode CLI sanity =="
cd "$OFFICIAL_ROOT/agentir"
uv run python -m tevatron.retriever.driver.encode --help > "$REPO/logs/tevatron_encode_help.txt" 2>&1 || {
  echo "FATAL: tevatron encode driver not importable; see logs/tevatron_encode_help.txt"; exit 4; }

echo "== 3. encode 4 shards in parallel (one per GPU) =="
PIDS=()
for i in 0 1 2 3; do
  if [ -f "$OUT/corpus.$i.pkl" ]; then echo "shard $i already done"; continue; fi
  CUDA_VISIBLE_DEVICES=$i uv run python -m tevatron.retriever.driver.encode \
    --output_dir "$TMPDIR/enc$i" \
    --model_name_or_path "$SNAP" \
    --tokenizer_name "$SNAP" \
    --bf16 \
    --pooling eos \
    --normalize \
    --per_device_eval_batch_size 64 \
    --passage_max_len 512 \
    --dataset_path "$TEV" \
    --dataset_number_of_shards 4 \
    --dataset_shard_index "$i" \
    --encode_output_path "$OUT/corpus.$i.pkl" \
    > "$REPO/logs/agentir_encode_shard$i.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
if [ "$FAIL" = 1 ]; then
  echo "FATAL: at least one encode shard failed; check logs/agentir_encode_shard*.log"
  echo "(if the failure is an unknown tevatron flag, fix flags against logs/tevatron_encode_help.txt)"
  exit 4
fi
ls -lh "$OUT"/corpus.*.pkl
echo "ALL_P2_AGENTIR_ENCODE_DONE"
submit_retry sbatch/p2_agentir.sbatch
