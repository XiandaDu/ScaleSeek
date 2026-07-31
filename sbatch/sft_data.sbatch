#!/usr/bin/env bash
# 阶段 1/3：SFT 数据构建 —— teacher 轨迹生成 + 转 verl parquet。
#
#   sbatch sbatch/sft_data.sbatch
#
# 覆盖项（sbatch --export=ALL,VAR=val 或先 export）：
#   SFT_QUESTIONS   问题 JSONL {id,question,golden_answers}  [$DATASETS/popqa/test.normalized.jsonl]
#   SFT_TEACHER     teacher 模型                              [Qwen/Qwen3.6-35B-A3B]
#   SFT_TP          vLLM tensor-parallel 度                    [2]
#   SFT_LIMIT       只取前 N 条（冒烟用）                     [无上限]
#   SFT_PARAM_POLICY heuristic|search|teacher|race            [heuristic]
#     race = (k1,b) 配置竞速：默认参数检索不到证据的 hop 会被其他配置救援，
#     每题多臂前向展开后按证据支持度选最优轨迹，败者存为 preference 数据。
#     teacher 调用量约为 heuristic 的 2.5-3.5 倍（先探测再放量）。
#
# 产物：
#   $SFT_TRAJ_DIR/trajectories.jsonl   teacher 轨迹
#   $SFT_TRAJ_DIR/sft_train.parquet    verl 训练输入
#
# 为什么 2 卡：Qwen3.6-35B-A3B 的 bf16 权重是 67 GiB。单张 H100 80G 放得下权重，
# 但只剩约 9 GiB 给 KV cache，32k context 下会频繁抢占。TP=2 稳得多。
#SBATCH --account=def-amirhk_gpu
#SBATCH --job-name=ss_sft_data
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/home/a32du/ScaleSeek/logs/%x-%j.out

source /home/a32du/ScaleSeek/sbatch/common.sh
TP="${SFT_TP:-2}"
require_full_gpus "$TP"
trap 'stop_vllm; cleanup_tmpdir' EXIT

# 默认用 RL 的训练集，不是 PopQA。PopQA 只有 test split，而 TASK.md 第二阶段要
# 在全部 14,267 条 PopQA 上评测，prepare_rl_data.py 还把 popqa 标为 out-of-domain。
# 拿它生成 SFT 冷启动数据 = 拿评测集训练，主实验作废。
# $RL_DATA_DIR/train.jsonl 是 NQ/HotpotQA/2Wiki/MuSiQue 的 *train* split，
# 与 RL 同源、无泄漏，也正是 generate_sft_data.py 文档里写的用法。
QUESTIONS="${SFT_QUESTIONS:-$RL_DATA_DIR/train.jsonl}"

# 语料/索引覆盖：长文档 regime（article 级 wiki-18，p0_articles.sbatch 的产物）
# 用 SFT_INDEX_DIR + SFT_CORPUS_FILE 指过来。CORPUS_FILE 同时驱动
# generate_sft_data 的 --param-grid auto 检测（长文档语料自动切换到含
# k1=16/25、b=1.0 的 Pi-Serini 区域网格）。
BM25_INDEX_DIR="${SFT_INDEX_DIR:-$BM25_INDEX_DIR}"
export CORPUS_FILE="${SFT_CORPUS_FILE:-$CORPUS_FILE}"
TEACHER_MODEL="${SFT_TEACHER:-Qwen/Qwen3.6-35B-A3B}"
POLICY="${SFT_PARAM_POLICY:-heuristic}"
# SFT_TAG 给产物加后缀，用于 A/B 探测时不互相覆盖（例：SFT_TAG=nostrict）
TAG="${SFT_TAG:+_$SFT_TAG}"
TRAJ=$SFT_TRAJ_DIR/trajectories${TAG}.jsonl
PARQUET=$SFT_TRAJ_DIR/sft_train${TAG}.parquet

# --strict 会跳过「每一跳都能在检索文档里验证」不成立的样本。2026-07-30 的 50 条
# 探测里它砍掉了 72%（问题集 61% 是多跳，逐跳验证对多跳很苛刻）。
# 设 SFT_STRICT=0 关掉，用于量化「良率损失 vs 数据质量」的取舍。
STRICT_FLAG="--strict"
[ "${SFT_STRICT:-1}" = "0" ] && STRICT_FLAG=""

# --- 前置检查：缺东西就早死，别跑 6 小时再发现索引是空的 ---
[ -f "$QUESTIONS" ] || { echo "FATAL: 问题文件不存在: $QUESTIONS
  先跑: \$PY scripts/download_data.py  (或设 SFT_QUESTIONS)"; exit 1; }
[ -d "$BM25_INDEX_DIR/index" ] || { echo "FATAL: BM25 索引不存在: $BM25_INDEX_DIR/index
  先跑: \$PY scripts/build_bm25_index.py"; exit 1; }

# 12h 墙钟是硬约束，且本作业不可续跑（轨迹一次性写出）。全量 202,553 条在 35B
# teacher 上远跑不完，超时会让 afterok 依赖把后面两级一起取消。默认限流；
# 先小批跑一次量出吞吐，再按 (12h × 余量) / 每条秒数 调大 SFT_LIMIT。
export SFT_LIMIT="${SFT_LIMIT:-3000}"
_avail=$(wc -l < "$QUESTIONS")
echo "[sft_data] questions=$QUESTIONS ($_avail 条可用，取前 $SFT_LIMIT)"
echo "[sft_data] teacher=$TEACHER_MODEL  tp=$TP  param-policy=$POLICY"
echo "[sft_data] -> $TRAJ"

# --- teacher 用 vLLM 服务，比 hf: 后端快一个量级 ---
# 35B 权重加载慢，超时给足（默认 2400s 对 67 GiB 从 /scratch 读可能不够）
start_vllm "$TEACHER_MODEL" main "$LLM_PORT" "$TP" teacher
waitp "$LLM_PORT" "${SFT_VLLM_TIMEOUT:-3600}"

"$PY" scripts/generate_sft_data.py \
    --questions "$QUESTIONS" \
    --out "$TRAJ" \
    --teacher "openai:http://127.0.0.1:$LLM_PORT/v1" \
    --teacher-model teacher \
    --index-dir "$BM25_INDEX_DIR" \
    --param-policy "$POLICY" \
    ${SFT_LIMIT:+--limit "$SFT_LIMIT"} \
    $STRICT_FLAG

stop_vllm

# --- 轨迹 -> verl parquet ---
"$PY" -m train.sft_dataset --in "$TRAJ" --out "$PARQUET"

echo "[sft_data] 完成:"
# 不要用 wc -l 报「产出多少」：--strict 会把跳过的样本也写进 jsonl（空 messages），
# 行数等于输入问题数，与可用轨迹数无关。2026-07-30 的探测里 50 行只有 7 条可用，
# 这个误导性输出害我把良率误判成 100%。以 summary 里的 kept_for_sft 为准。
if [ -s "${TRAJ%.jsonl}.summary.json" ]; then
  "$PY" -c "
import json,sys
d=json.load(open('${TRAJ%.jsonl}.summary.json'))
k=d.get('kept_for_sft',0); q=d.get('questions',0)
print(f'  可用轨迹 (kept_for_sft): {k} / {q} 个问题  = 良率 {100*k/max(q,1):.0f}%')
print(f'  分状态: {d.get(\"counts\")}')
"
else
  echo "  warn: 没有 summary.json，无法报告良率"
fi
ls -lh "$PARQUET"

# ── 参数多样性闸门 ───────────────────────────────────────────────────────────
# 防两种已发生过的数据退化（reports/param_policy_findings.md）：
#   1. 参数缺失 —— teacher 天然不设 k1/b/top_k（4B/8B 均 0/N 次）；
#   2. 参数坍缩成常数 —— heuristic 学生曾输出固定 1.5/0.75/5。
# 任一成立则 SFT 教不出参数控制，下游 24h SFT + 24h RL 的自适应论点直接作废，
# 所以在数据阶段硬失败，而不是训练完才发现。
# 判据（2026-07-30 修订）：ok 轨迹的 bm25 调用 >=95% 带参数即可。
# 所有参数（含 top_k）恒定都只 warn 不拦 —— search 探测证明了结果为真的常数
# 不是退化：strict 成功过滤 + 结果驱动选参在 wiki-18 上必然收敛到常数
# （top_k=3 13/13）。自适应的教学载体改为失败->恢复回合（inject_failures），
# 这里统计其覆盖率；ok>=20 却零恢复回合时大声警告（自适应论点无数据支撑）。
"$PY" - "$TRAJ" <<'PYEOF' || { echo "FATAL: 参数多样性闸门未通过"; exit 1; }
import collections, json, sys
n_calls = n_with = n_ok = n_recovery = 0
vals = {"top_k": collections.Counter(), "k1": collections.Counter(),
        "b": collections.Counter(), "mode": collections.Counter()}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("status") != "ok":
        continue
    n_ok += 1
    if (r.get("meta") or {}).get("n_failure_turns", 0) > 0:
        n_recovery += 1
    for m in r["messages"]:
        if m["role"] != "assistant" or "bm25_retrieve" not in m["content"]:
            continue
        seg = m["content"].split("<tool_call>", 1)[-1].split("</tool_call>", 1)[0]
        try:
            args = json.loads(seg).get("arguments", {})
        except Exception:
            continue
        n_calls += 1
        if all(k in args for k in ("k1", "b", "top_k")):
            n_with += 1
        for k in vals:
            if k in args:
                vals[k][args[k]] += 1
print(f"[param-gate] bm25 调用 {n_calls} 次，带完整参数 {n_with} 次")
for k, c in vals.items():
    print(f"[param-gate]   {k}: {dict(c.most_common(6))}")
if n_calls == 0:
    sys.exit("param-gate FAIL: ok 轨迹里没有任何 bm25 调用")
# 95% 阈值而非零容忍：闸门要抓的是系统性退化（teacher 全程不设参数那种），
# 不是个别边角。零容忍曾有让 24h 作业在最后一步因 1 个调用翻车的风险
# （_search_params 的旧 fallback 会发裸 mode 调用；根因已修，此处纵深防御）。
if n_with < n_calls * 95 // 100:
    sys.exit(f"param-gate FAIL: {n_calls - n_with}/{n_calls} 次调用缺参数（>5%）—— 系统性退化")
if n_with < n_calls:
    print(f"[param-gate] warn: {n_calls - n_with} 次调用缺参数（<=5%，容忍）")
print(f"[param-gate] 失败->恢复回合: {n_recovery}/{n_ok} 条 ok 轨迹")
for k in ("top_k", "k1", "b"):
    if len(vals[k]) < 2:
        print(f"[param-gate] warn: {k} 恒定（结果为真的常数，见 findings addendum）")
if n_ok >= 20 and n_recovery == 0:
    print("[param-gate] WARN(响亮): 零失败->恢复回合 —— 自适应行为在这批数据里没有任何示范，")
    print("[param-gate]              trained-adaptive-policy 论点将完全依赖 RL 探索。")
PYEOF

echo "下一步: sbatch sbatch/sft_train.sbatch"
