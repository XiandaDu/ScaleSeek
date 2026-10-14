"""Corpora: one materialised JSONL per corpus, shared by every retriever (BASELINES.md §3).

paths.data/corpora/<name>/
    corpus.jsonl          {"id", "contents"} -- THE document text for bm25, e5, qwen3_emb_4b
    offsets.npy           uint64 byte offset of every line (+ end), for O(1) lookups
    ids.json              doc id per line (omitted when id == line number, as in Wiki-18)
    tevatron.jsonl        {"docid", "text"} -- the same text, in the format the official
                          AgentIR/tevatron encoder and searcher read
    grepseek/wiki_corpus.jsonl   hard link to corpus.jsonl; the ONLY file in GrepSeek's
                          working directory (the harness requires this name)
    bm25_input/corpus.jsonl      hard link; Anserini indexes every file in its input dir
    manifest.json         source pin, raw SHA256, doc count, bytes, SHA256s, id rule

Smoke corpora (paths.smoke/corpora/<name>_mini) have the same layout and keep the ORIGINAL
doc ids, so a smoke trajectory can be checked against the full corpus.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from .hub import snapshot_dir
from .utils import PipelineError, log, now_iso, read_json, sha256_file, sha256_text, write_json


def corpus_dir(base: dict, name: str, smoke: bool = False) -> Path:
    if smoke:
        return Path(base["paths"]["smoke"]) / "corpora" / f"{name}_mini"
    return Path(base["paths"]["data"]) / "corpora" / name


# ============================================================================= display
def split_title(contents: str, rule: str) -> tuple[str, str]:
    """Title and body of a document for DISPLAY in our own prompt formats.  The rule is
    the corpus's `title_rule` (base.yaml); `first_line` is the Search-R1 convention."""
    if rule == "first_line":
        head, _, rest = contents.partition("\n")
        return head, rest
    if rule == "bcp_front_matter":
        # BrowseComp-Plus pages: "---\ntitle: T\ndate: D\n---\nbody"
        if contents.startswith("---\n"):
            end = contents.find("\n---\n", 4)
            if end != -1:
                front, body = contents[4:end], contents[end + 5:]
                for line in front.split("\n"):
                    if line.startswith("title:"):
                        return line[len("title:"):].strip(), body
        head, _, rest = contents.partition("\n")
        return head, rest
    raise PipelineError(f"unknown title_rule {rule!r}")


# ============================================================================= materialise
def _extract_wiki18(raw: Path, dst: Path) -> str:
    """wiki-18.jsonl.gz is a gzipped TAR whose member is the JSONL (see GrepSeek
    sft/data_generation/download_corpus.py)."""
    with tarfile.open(raw, "r|gz") as tf:
        for member in tf:
            if member.name.endswith(".jsonl"):
                with tf.extractfile(member) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out, length=64 << 20)
                return f"tar member {member.name}"
    raise PipelineError(f"{raw}: tar without a .jsonl member")


def _write_bcp_corpus(files: list[str], dst: Path) -> str:
    import pyarrow.parquet as pq

    with open(dst, "w", encoding="utf-8") as out:
        for f in files:
            for rec in pq.read_table(f, columns=["docid", "text"]).to_pylist():
                out.write(json.dumps({"id": str(rec["docid"]), "contents": rec["text"]},
                                     ensure_ascii=False) + "\n")
    return f"{len(files)} parquet shards"


def _index_lines(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """One pass: byte offsets, ids, and the structural checks of the preflight."""
    offsets, ids, bad = [0], [], []
    empty = 0
    with open(path, "rb") as f:
        pos = 0
        for n, line in enumerate(f):
            pos += len(line)
            offsets.append(pos)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad.append(n)
                ids.append("")
                continue
            if set(obj) != {"id", "contents"} or not isinstance(obj["contents"], str):
                bad.append(n)
            if not obj.get("contents", "").strip():
                empty += 1
            ids.append(str(obj.get("id")))
    return np.asarray(offsets, dtype=np.uint64), ids, {"bad_lines": bad[:20], "n_bad_lines": len(bad),
                                                        "n_empty_contents": empty}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def _finish_corpus_dir(out: Path, expected: int | None) -> dict:
    """Shared tail of full and smoke materialisation: index, checks, derived files."""
    corpus = out / "corpus.jsonl"
    offsets, ids, checks = _index_lines(corpus)
    n = len(ids)
    uniq = len(set(ids))
    problems = []
    if expected is not None and n != expected:
        problems.append(f"{n} docs != expected {expected}")
    if uniq != n:
        problems.append(f"{n - uniq} duplicate doc ids")
    if checks["n_bad_lines"]:
        problems.append(f"{checks['n_bad_lines']} malformed lines, e.g. {checks['bad_lines'][:5]}")
    if problems:
        raise PipelineError(f"corpus check FAILED for {out}: " + "; ".join(problems))
    np.save(out / "offsets.npy", offsets)
    id_is_line = all(i == str(k) for k, i in enumerate(ids))
    if not id_is_line:
        write_json(out / "ids.json", ids)
    with open(corpus, "r", encoding="utf-8") as src, open(out / "tevatron.jsonl", "w", encoding="utf-8") as dst:
        for line in src:
            obj = json.loads(line)
            dst.write(json.dumps({"docid": obj["id"], "text": obj["contents"]}, ensure_ascii=False) + "\n")
    _link(corpus, out / "grepseek" / "wiki_corpus.jsonl")
    _link(corpus, out / "bm25_input" / "corpus.jsonl")
    return {"docs": n, "bytes": int(offsets[-1]), "sha256": sha256_file(corpus),
            "id_is_line_index": id_is_line, "ids_sha256": sha256_text("\n".join(ids)),
            "tevatron_sha256": sha256_file(out / "tevatron.jsonl"), **checks}


def prepare_corpus(base: dict, name: str) -> dict:
    cfg = base["corpora"][name]
    out = corpus_dir(base, name)
    out.mkdir(parents=True, exist_ok=True)
    snap = snapshot_dir(base, cfg["hf_repo"], cfg["revision"], "dataset")
    tmp = out / "corpus.jsonl.part"
    if cfg["source"] == "wiki18":
        raw = snap / cfg["file"]
        raw_sha = {raw.name: sha256_file(raw)}
        how = _extract_wiki18(raw, tmp)
    else:                                       # browsecomp_plus_corpus
        files = sorted(glob.glob(str(snap / cfg["file"])))
        raw_sha = {Path(f).name: sha256_file(f) for f in files}
        how = _write_bcp_corpus(files, tmp)
    os.replace(tmp, out / "corpus.jsonl")
    info = _finish_corpus_dir(out, cfg["expected_docs"])
    manifest = {"corpus": name, "created": now_iso(), "smoke": False,
                "source": {"hf_repo": cfg["hf_repo"], "revision": cfg["revision"],
                           "file": cfg["file"], "raw_sha256": raw_sha, "materialised_from": how},
                "title_rule": cfg["title_rule"], **info}
    write_json(out / "manifest.json", manifest)
    log(f"corpus ok: {name} docs={info['docs']} sha256={info['sha256'][:12]}")
    return manifest


def load_corpus_manifest(base: dict, name: str, smoke: bool = False) -> dict:
    return read_json(corpus_dir(base, name, smoke) / "manifest.json")


# ============================================================================= smoke corpus
def make_mini_corpus(base: dict, name: str, head_docs: int, must_include_rows: list[int]) -> dict:
    """A small corpus = the first `head_docs` documents + the rows that must be present
    (documents containing the smoke questions' gold answers / BCP evidence)."""
    full = corpus_dir(base, name)
    parent = load_corpus_manifest(base, name)
    rows = sorted(set(range(min(head_docs, parent["docs"]))) | set(must_include_rows))
    out = corpus_dir(base, name, smoke=True)
    if (out / "manifest.json").exists():
        old = read_json(out / "manifest.json")
        if (old["parent"]["sha256"] == parent["sha256"] and old["head_docs"] == head_docs
                and old["extra_rows"] == sorted(set(must_include_rows))):
            log(f"mini corpus {name}_mini exists and matches")
            return old
    out.mkdir(parents=True, exist_ok=True)
    store = DocStore(full)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(store.get_row(r), ensure_ascii=False) + "\n")
    info = _finish_corpus_dir(out, None)
    manifest = {"corpus": f"{name}_mini", "created": now_iso(), "smoke": True,
                "parent": {"corpus": name, "sha256": parent["sha256"], "docs": parent["docs"]},
                "head_docs": head_docs, "extra_rows": sorted(set(must_include_rows)),
                "title_rule": parent["title_rule"], **info}
    write_json(out / "manifest.json", manifest)
    log(f"mini corpus: {name}_mini docs={info['docs']}")
    return manifest


def grep_rows(corpus_file: Path, needles: list[str], per_needle: int) -> list[int]:
    """0-based line numbers of the first `per_needle` lines containing each needle (rg -F)."""
    rows: set[int] = set()
    for needle in needles:
        if not needle.strip():
            continue
        res = subprocess.run(["rg", "-F", "-n", "-m", str(per_needle), "--no-config", "--", needle,
                              str(corpus_file)], capture_output=True, text=True)
        if res.returncode not in (0, 1):
            raise PipelineError(f"rg failed on {needle!r}: {res.stderr[:200]}")
        for line in res.stdout.splitlines():
            rows.add(int(line.split(":", 1)[0]) - 1)
    return sorted(rows)


# ============================================================================= docstore
class DocStore:
    """Random access to corpus.jsonl by doc id or line number (os.pread: thread-safe)."""

    def __init__(self, cdir: Path):
        cdir = Path(cdir)
        self.dir = cdir
        self.manifest = read_json(cdir / "manifest.json")
        self.offsets = np.load(cdir / "offsets.npy", mmap_mode="r")
        self.n = len(self.offsets) - 1
        if self.manifest["id_is_line_index"]:
            self.ids = None
            self._row_of = None
        else:
            self.ids = read_json(cdir / "ids.json")
            self._row_of = {d: i for i, d in enumerate(self.ids)}
        self._fd = os.open(cdir / "corpus.jsonl", os.O_RDONLY)

    def row_of(self, doc_id: str) -> int:
        return int(doc_id) if self._row_of is None else self._row_of[doc_id]

    def __contains__(self, doc_id: str) -> bool:
        return 0 <= int(doc_id) < self.n if self._row_of is None else doc_id in self._row_of

    def id_of(self, row: int) -> str:
        return str(row) if self.ids is None else self.ids[row]

    def get_row(self, row: int) -> dict:
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return json.loads(os.pread(self._fd, end - start, start))

    def get(self, doc_id: str) -> dict:
        return self.get_row(self.row_of(doc_id))

    def iter_rows(self, start: int, stop: int):
        """Sequential read of rows [start, stop) -- for encoding shards."""
        with open(self.dir / "corpus.jsonl", "rb") as f:
            f.seek(int(self.offsets[start]))
            for _ in range(start, stop):
                yield json.loads(f.readline())
