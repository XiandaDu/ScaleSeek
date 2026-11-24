"""GrepSeek+ tool layer against the real corpus (no LLM)."""
import json, sys, time
sys.path.insert(0, "/scratch/a32du/scaleseek_v2/repos/grepseek")
sys.path.insert(1, "/home/a32du/ScaleSeek/scaleseek")
from inference import tools
import grepseek_plus as gp
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/scratch/a32du/scaleseek_v2/hf/hub/models--alireza7--GrepSeek-Qwen3.5-9B-GRPO/snapshots/a79563970cfdd2ced3cc5fde481737d0ebea6fa4")
corpus_dir = sys.argv[1]
KW = dict(corpus_dir=corpus_dir, timeout=60, max_tokens=2048, tokenizer=tok)


def plus(cmd, variants, turn=1, used=1000):
    gp.CFG.update(variants=set(variants), context=16384)
    gp.S.turn, gp.S.used, gp.S.turns = turn, used, 6
    t0 = time.time()
    r = gp.run_tool(cmd, **KW)
    return json.loads(r.to_payload()), time.time() - t0


def show(tag, p, dt):
    lines = [ln for ln in p["stdout"].split("\n") if ln.strip()]
    titles = [ln.split('"contents": "\\"', 1)[1].split('\\"', 1)[0] if '"contents": "\\"' in ln else ln[:60] for ln in lines]
    print(f"  {tag:22s} {dt:5.1f}s  {len(lines)} lines  exit {p['exit_code']}  truncated {p['truncated']}  keys {len(p)}"
          f"  notice: {p.get('notice')}\n      titles: {titles[:8]}")


same = 0
CMDS = ['rg -F "Evan Morris" corpus.jsonl | head -n 3', 'rg -F "Charles Darwin" corpus.jsonl | head -n 8',
        'rg -F "zzqqxx" corpus.jsonl | head -n 3', 'rg -F "Paris" corpus.jsonl | rg -i -F "capital" | head -n 8',
        'rg -F "Paris" corpus.jsonl | wc -l', 'cat /etc/passwd', 'rg -F "Darwin" corpus.jsonl | head -n 40']
for cmd in CMDS:
    off = tools.run_tool(cmd, **KW).to_payload()
    same += off == gp.run_tool.__wrapped__(cmd, **KW).to_payload() if hasattr(gp.run_tool, "__wrapped__") else off == json.dumps(plus(cmd, [])[0], ensure_ascii=False)
print(f"no variants == official payload: {same}/{len(CMDS)}")

for cmd, variants in [
    ('rg -F "Jonas Salk" corpus.jsonl | rg -i -F "religion" | head -n 3', ["title"]),
    ('rg -F "Mount Pinatubo" corpus.jsonl | rg -i -F "eruption" | rg -i -F "death toll of 847" | head -n 4', ["relax", "title"]),
    ('rg -F "Crash" corpus.jsonl | rg -i -F "film" | rg -i -F "Haggis" | rg -i -F "zzqq" | head -n 6', ["relax", "title"]),
    ('rg -F "the Naked angel" corpus.jsonl | rg -i -F "directed" | head -n 3', ["fold", "relax", "title"]),
    ('rg -F "Paris" corpus.jsonl | rg -i -F "capital" | head -n 5', ["fold", "relax", "title"]),
    ('rg -F "zzqqxx" corpus.jsonl | rg -F "abc" | head -n 3', ["fold", "relax", "title"]),
]:
    print(cmd)
    t0 = time.time(); off = json.loads(tools.run_tool(cmd, **KW).to_payload()); show("official", off, time.time() - t0)
    show("+" + ",".join(variants), *plus(cmd, variants))

