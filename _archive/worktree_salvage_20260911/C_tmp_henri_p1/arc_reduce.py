#!/usr/bin/env python
"""Deterministic per-env reduction of ARC stage JSONL telemetry + scorecards.
Usage: python arc_reduce.py <telemetry_root> [stage...]
Output: one JSON line per env file (telemetry stats) + one per scorecard.
Handles nested ARC_EPISODE_TRACE events (tele.emit envelope).
"""
import json, glob, os, sys, math
from collections import Counter

def entropy(cnt):
    tot = sum(cnt.values())
    if tot <= 1:
        return 0.0
    return max(0.0, -sum((c / tot) * math.log2(c / tot) for c in cnt.values() if c > 0))

def summarize_jsonl(path):
    recs, trace = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("schema_id") == "henri.arc-episode-trace.v1":
            trace = d
        elif d.get("event_type") == "ARC_EPISODE_TRACE" and isinstance(d.get("trace"), dict):
            trace = d["trace"]
        else:
            recs.append(d)
    out = {"file": os.path.basename(path)}
    if trace:
        for k in ("env", "policy", "learning_frozen", "diagnostic_only", "terminal_state",
                  "exact_pass", "steps_run", "veto_count", "planning_ms", "scorecard_id",
                  "task_input_sha256", "levels_completed", "commit_sha256",
                  "demo_pair_count", "candidate_count", "action_entropy", "min_sagnac_delta",
                  "external_state_delta", "evaluator_status"):
            if k in trace:
                out["trace_" + k] = trace[k]
    acts = Counter(r.get("action") for r in recs if r.get("action"))
    out["n_records"] = len(recs)
    out["n_actions"] = sum(acts.values())
    out["action_entropy"] = round(entropy(acts), 4)
    out["action_top3"] = [a for a, _ in acts.most_common(3)]
    for k in ("sagnac_delta", "free_energy", "kuramoto_r", "efe_best", "efe_spread",
              "valence", "grid_dist", "step_ms", "goal_distance"):
        vals = [r[k] for r in recs if isinstance(r.get(k), (int, float))]
        if vals:
            out[k + "_min"] = round(min(vals), 4)
            out[k + "_mean"] = round(sum(vals) / len(vals), 4)
            out[k + "_max"] = round(max(vals), 4)
    tl = [r["transition_loss"] for r in recs if isinstance(r.get("transition_loss"), (int, float))]
    if tl:
        out["transition_loss_n"] = len(tl)
        out["transition_loss_mean"] = round(sum(tl) / len(tl), 4)
        out["transition_loss_first"] = round(tl[0], 4)
        out["transition_loss_last"] = round(tl[-1], 4)
    le = [r["loss_ema"] for r in recs if isinstance(r.get("loss_ema"), (int, float))]
    if le:
        out["loss_ema_first"] = round(le[0], 4)
        out["loss_ema_last"] = round(le[-1], 4)
    rh = [r["recall"] for r in recs if isinstance(r.get("recall"), dict)]
    if rh:
        out["recall_hits_total"] = sum(r.get("hits", 0) for r in rh)
        sims = [r.get("top_sim", 0.0) for r in rh if isinstance(r.get("top_sim"), (int, float))]
        if sims:
            out["recall_top_sim_max"] = round(max(sims), 4)
    vr = [r["action_embedding_divergence"] for r in recs
          if isinstance(r.get("action_embedding_divergence"), (int, float))]
    if vr:
        out["action_emb_div_first"] = round(vr[0], 4)
        out["action_emb_div_last"] = round(vr[-1], 4)
    return out

def summarize_scorecard(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return {"file": os.path.basename(path), "parse_error": str(e)}
    out = {"file": os.path.basename(path)}
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        out["env"] = k
        out["score"] = v.get("score")
        out["card_id"] = v.get("card_id")
        txt = str(v)
        # Extract the LAST EnvironmentScore entry (canonical run; first is a
        # ghost id=None entry emitted by the arcade API).
        entries = txt.split("EnvironmentScore(")
        if len(entries) > 1:
            last = entries[-1]
            for pat in ("score=", "levels_completed=", "actions=", "state="):
                idx = last.find(pat)
                if idx >= 0:
                    seg = last[idx + len(pat):]
                    val = seg.split(",")[0].split("]")[0].split(")")[0]
                    out["run_" + pat.rstrip("=")] = val
        b = txt.find("level_baseline_actions=[")
        if b >= 0:
            out["level_baseline_actions_head"] = txt[b + len("level_baseline_actions=["):b + len("level_baseline_actions=[") + 20]
    return out

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    stages = sys.argv[2:] or ["s3", "s4", "s5"]
    for stage in stages:
        base = os.path.join(root, stage)
        if not os.path.isdir(base):
            continue
        for cell in sorted(os.listdir(base)):
            cbase = os.path.join(base, cell)
            if not os.path.isdir(cbase):
                continue
            for env in sorted(os.listdir(cbase)):
                ebase = os.path.join(cbase, env)
                if not os.path.isdir(ebase):
                    continue
                for f in sorted(glob.glob(os.path.join(ebase, "*.jsonl"))):
                    print(json.dumps(summarize_jsonl(f)))
                for f in sorted(glob.glob(os.path.join(ebase, "*scorecards*.json"))):
                    print(json.dumps(summarize_scorecard(f)))
