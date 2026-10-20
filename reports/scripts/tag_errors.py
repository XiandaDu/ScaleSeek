"""Search-o1 malformed search tags.  A main-model turn that writes a search query must end on the stop
string <|end_search_query|>; count turns that mention `search_query` but did not stop on it (a
misspelt tag: `|end_search_query|>`, `<|end_search_query>`, ...) and what became of those questions.
Usage: tag_errors.py [run_dir ...]  (default: every formal Search-o1 run).
Writes analysis/failures/tag_errors.json when run on the formal runs."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/scratch/a32du/scaleseek_v2")
END = "<|end_search_query|>"


def profile(path: Path) -> dict:
    n = turns = bad_turns = bad_q = bad_first = 0
    finish, bad_finish, shapes = Counter(), Counter(), Counter()
    for line in open(path):
        r = json.loads(line)
        n += 1
        finish[r["finish_reason"]] += 1
        main = [{**t, "text": t.get("text") or ""} for t in r["trajectory"] if t["type"] == "llm" and "role" not in t]
        bad = [t for t in main if "search_query" in t["text"] and t["stop_reason"] != END]
        turns += len(main)
        bad_turns += len(bad)
        if bad:
            bad_q += 1
            bad_first += bad[0] is main[0]
            bad_finish[r["finish_reason"]] += 1
            tail = bad[0]["text"].rsplit("search_query", 1)[1][:3]
            shapes[bad[0]["text"].rsplit("search_query", 1)[0][-5:] + "search_query" + tail] += 1
    return {"n": n, "finish": finish, "main_turns": turns, "bad_turns": bad_turns, "questions_with_bad_tag": bad_q,
            "bad_on_first_turn": bad_first, "finish_of_bad_questions": bad_finish, "shapes": shapes.most_common(6)}


if __name__ == "__main__":
    dirs = [Path(p) for p in sys.argv[1:]] or sorted(ROOT.glob("runs/*/search_o1__*"))
    out = {f"{d.parent.name}/{d.name}": profile(d / "trajectories.jsonl") for d in dirs}
    if not sys.argv[1:]:
        (ROOT / "analysis" / "failures" / "tag_errors.json").write_text(json.dumps(out, indent=1, default=dict))
    for k, v in out.items():
        print(f"{k:45s} n={v['n']} no_answer={v['finish'].get('no_answer', 0):3d} bad_q={v['questions_with_bad_tag']:3d} "
              f"bad_turns={v['bad_turns']}/{v['main_turns']} first={v['bad_on_first_turn']} -> {dict(v['finish_of_bad_questions'])} {v['shapes'][:3]}")
