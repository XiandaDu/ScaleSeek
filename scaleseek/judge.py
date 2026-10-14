"""BrowseComp-Plus judge (BASELINES.md §4): one independent server of the judge model
(configs/methods/bcp_judge.yaml; Qwen3-32B since 2026-10-14).

For every record of a BCP run: no final answer, or a run that did not complete, is
judged incorrect WITHOUT a model call (official evaluate_run.py does the same).  Otherwise
the official create_judge_prompt(question, response=<final answer>, correct_answer) goes
to the judge as one user message, thinking disabled exactly as the official script does
(chat_template_kwargs={"enable_thinking": False}), temperature 0, top_p 1, at most
`max_new_tokens`; the verdict is the official parse_judge_response.  The judge never sees
the agent's conversation.  Raw judge text is stored in <run_dir>/judge__<judge>.jsonl;
verdicts of another judge model under the same name are refused, never mixed or resumed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ConfigError, load_method, load_prompts
from .hub import local_snapshot
from .llm import LLMClient, LLMRequestError, VLLMServer
from .official import verify_method
from .runner import job_gpus
from .utils import (JsonlAppender, PipelineError, git_state, iter_jsonl, log, now_iso, repo_root,
                    write_json)


def judge_file(run_dir: Path, judge_name: str) -> Path:
    """One file per judge configuration, so a different judge (e.g. a larger model)
    never overwrites earlier verdicts."""
    return run_dir / f"judge__{judge_name}.jsonl"


def judge_runs(base: dict, run_dirs: list[Path], judge_name: str) -> None:
    cfg = load_method(judge_name, base)
    if cfg["method"] != "bcp_judge":
        raise ConfigError(f"{judge_name} is not a judge configuration")
    prompts = load_prompts(cfg)
    official = verify_method(base, cfg)
    for run_dir in run_dirs:             # before loading the model: fail fast
        out = judge_file(run_dir, judge_name)
        if out.exists() and any(r["judge_model"] != cfg["generator"] for r in iter_jsonl(out)):
            raise PipelineError(f"{out} holds verdicts of another judge model: move it away first")
    model_path = local_snapshot(base, cfg["generator"]["model"], cfg["generator"]["revision"])
    server = VLLMServer(model_path, cfg["server"], cfg["server"]["max_model_len"],
                        run_dirs[0].parent / "judge_vllm.log", job_gpus())
    dec = cfg["decoding"]
    try:
        server.start(base["runtime"]["server_start_timeout_s"])
        client = LLMClient(server, cfg["client"])
        for run_dir in run_dirs:
            out = judge_file(run_dir, judge_name)
            done = {r["id"] for r in iter_jsonl(out)} if out.exists() else set()
            todo = [r for r in iter_jsonl(run_dir / "trajectories.jsonl") if r["id"] not in done]
            log(f"judge {run_dir.name}: {len(done)} done, {len(todo)} to judge")

            def one(rec: dict) -> dict:
                row = {"id": rec["id"], "judge_model": cfg["generator"], "created": now_iso()}
                # The final response if the method has one (AgentIR: the official
                # Explanation/Exact Answer/Confidence text), else its completed answer.
                response = rec.get("final_response") or (
                    rec["prediction"] if rec["finish_reason"] == "answer" else None) or ""
                if not response:
                    return {**row, "response": response, "judge_prompt": None,
                            "judge_response": None, "called": False,
                            "judge_result": {"correct": False, "parse_error": False,
                                             "reason": "no completed final answer"}}
                prompt = prompts["create_judge_prompt"](rec["question"], response,
                                                        rec["golden_answers"][0])
                try:
                    out_ = client.chat([{"role": "user", "content": prompt}],
                                       temperature=dec["temperature"], top_p=dec["top_p"],
                                       max_tokens=dec["max_new_tokens"],
                                       chat_template_kwargs={"enable_thinking": False})
                    text, err = out_["content"], None
                except LLMRequestError as exc:
                    text, err = "", str(exc)
                return {**row, "response": response, "judge_prompt": prompt,
                        "judge_response": text, "called": True, "api_error": err,
                        "judge_result": prompts["parse_judge_response"](text)}

            w = JsonlAppender(out)
            try:
                with ThreadPoolExecutor(cfg["client"]["concurrency"]) as pool:
                    for row in pool.map(one, todo):
                        w.write(row)
            finally:
                w.close()
            write_json(run_dir / f"judge__{judge_name}.provenance.json",
                       {"judge_config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                        "harness_commit": git_state(repo_root())["commit"],
                        "official": official, "server_command": server.command()})
    finally:
        server.stop()
