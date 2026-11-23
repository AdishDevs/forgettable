"""Rerun only the unlearning step on the stored test adapters with the settings in tuning/selected.json.

For each persona: starts from the stored theta adapter (no retraining), runs every method with its
selected (lr, T, alpha), evaluates log-probabilities as in pilot.py, and adds greedy decoding on the
held-out probes for theta, the retrain and every method. Writes persona_{pi}_tuned.json next to the
existing results (never overwrites them).
Usage: .venv/bin/python rerun_tuned.py results 0 20 0
"""
import json, random, sys, time
import mlx.core as mx
from mlx_lm import load
import pilot

METHODS = ["GradDiff", "NPO", "SimNPO", "STR", "ReplayOnly"]
MAX_NEW = 8


def greedy(model, tok, question):
    msgs = [{"role": "system", "content": pilot.SYS}, {"role": "user", "content": question}]
    ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
    out = []
    for _ in range(MAX_NEW):
        nxt = int(mx.argmax(model(mx.array(ids + out)[None])[0, -1]).item())
        if nxt in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
            break
        out.append(nxt)
    return tok.decode(out).strip()


def greedy_eval(model, tok, facts):
    res = []
    for f in facts:
        answers = [greedy(model, tok, q) for q in pilot.SLOTS[f["slot"]][3]]
        low = [a.lower() for a in answers]
        res.append({"answers": answers,
                    "o": sum(f["o"].lower() in a for a in low),
                    "n": sum(bool(f["n"]) and f["n"].lower() in a for a in low)})
    return res


def main():
    rdir, start, end, seed = pilot.HERE / sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    sel = json.load(open(pilot.HERE / "tuning" / "selected.json"))["selected"]
    for pi in range(start, end):
        out = rdir / f"persona_{pi}_tuned.json"
        if out.exists():
            continue
        facts, sessions, deleted = pilot.make_persona(random.Random(pi))
        d = json.load(open(rdir / f"persona_{pi}.json"))
        assert [f["o"] for f in facts] == [f["o"] for f in d["facts"]], "persona mismatch"
        retained = [s for s in sessions if s["id"] not in deleted]
        forget_s = [s for s in sessions if s["id"] in deleted]
        sup_ids = {s["id"] for s in retained
                   if any(f["role"] == "superseded" and f["slot"] == s["slot"] and s["value"] == f["o"] for f in facts)}
        gen, ev, times = {}, {}, {}
        for k in ("theta", "retrain"):
            model, tok = load(pilot.BASE, adapter_path=str(rdir / f"p{pi}" / k))
            gen[k] = greedy_eval(model, tok, facts)
        for m in METHODS:
            s = sel[m]
            model, tok = load(pilot.BASE, adapter_path=str(rdir / f"p{pi}" / "theta"))
            t0 = time.time()
            pilot.unlearn(model, tok, m, pilot.qa_items(tok, forget_s), pilot.qa_items(tok, retained), sup_ids,
                          random.Random(pi * 7 + 1 + 1000 * seed), T=s["T"], lr=s["lr"], alpha=s["alpha"])
            times[m] = time.time() - t0
            ev[m] = pilot.evaluate(model, tok, facts)
            gen[m] = greedy_eval(model, tok, facts)
        json.dump({"persona": pi, "seed": seed, "settings": {m: sel[m] for m in METHODS},
                   "eval": ev, "greedy": gen, "times": times}, open(out, "w"), indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] {rdir.name} persona {pi}: tuned rerun saved", flush=True)


if __name__ == "__main__":
    main()
