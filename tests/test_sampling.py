import json

from scaleseek import data


def _base(tmp_path, n_rows, cap, seed=7):
    base = {"paths": {"data": str(tmp_path)}, "sampling": {"seed": seed, "max_questions": cap}}
    d = data.dataset_dir(base, "toy")
    d.mkdir(parents=True)
    rows = [{"id": f"q{i}", "question": f"Q{i}?", "golden_answers": ["a"]} for i in range(n_rows)]
    (d / "normalized.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    from scaleseek.utils import sha256_file
    (d / "preflight.json").write_text(json.dumps({
        "ok": True, "problems": [], "normalized_sha256": sha256_file(d / "normalized.jsonl"),
        "source": {"hf_repo": "x", "revision": "y", "file": "f", "split": "test", "raw_sha256": "z"}}))
    return base, rows


def test_subset_is_order_independent_and_capped(tmp_path):
    base, rows = _base(tmp_path, 50, 10)
    subset, man = data.compute_subset(base, "toy")
    assert man["n"] == 10 and len(subset) == 10
    # dataset order preserved
    idx = [int(r["id"][1:]) for r in subset]
    assert idx == sorted(idx)
    # the choice depends only on (seed, id): shuffling the file changes nothing
    ranked = sorted((r["id"] for r in rows), key=lambda q: data._rank_key(7, q))[:10]
    assert set(ranked) == {r["id"] for r in subset}


def test_small_dataset_is_used_in_full(tmp_path):
    base, rows = _base(tmp_path, 8, 10)
    subset, man = data.compute_subset(base, "toy")
    assert [r["id"] for r in subset] == [r["id"] for r in rows]
    assert man["rule"].startswith("full")


def test_subset_never_overwritten(tmp_path):
    base, _ = _base(tmp_path, 30, 10)
    data.make_subset(base, "toy")
    data.make_subset(base, "toy")          # verifies, does not rewrite
    rows, man = data.load_subset(base, "toy")
    assert len(rows) == 10
