"""Forgettable pilot study: does approximate unlearning restore superseded values?

Pipeline per persona:
  1. generate sessions with ground-truth provenance
  2. train LoRA adapter theta on all sessions          (mlx_lm lora CLI)
  3. retrain reference R on sessions minus S           (same protocol)
  4. unlearn S from theta with GradDiff, NPO, SimNPO, STR (custom MLX loop)
  5. score reversion fidelity (RF), forgetting efficacy (FE) and utility
Usage: .venv/bin/python pilot.py --personas 4 --seed 0
"""
import argparse, json, math, os, random, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
SYS = "You are a personal assistant that remembers what the user told you. Answer with only the value."
HERE = Path(__file__).resolve().parent

# slot: (statement template, update template, training question, 3 held-out evaluation paraphrases, value pool)
SLOTS = {
    "city": ("I live in {v}.", "I have moved. I now live in {v}.", "Which city do I live in?",
             ["Remind me, which city is my home?", "Where am I living these days?", "What city do I live in right now?"],
             ["Colombo", "Kandy", "Galle", "Jaffna", "Negombo", "Matara", "Kurunegala", "Badulla"]),
    "employer": ("I work at {v}.", "I changed jobs. I now work at {v}.", "Where do I work?",
                 ["Which company employs me?", "Who is my employer?", "What company do I work for now?"],
                 ["Dialog", "Virtusa", "Hayleys", "Cargills", "Keells", "Sysco", "Brandix", "Hemas"]),
    "pet": ("My dog is called {v}.", "My old dog passed away. My new dog is called {v}.", "What is my dog's name?",
            ["What do I call my dog?", "Remind me of my dog's name.", "What is the name of my dog?"],
            ["Bruno", "Luna", "Rocky", "Coco", "Milo", "Simba", "Bella", "Tiger"]),
    "sport": ("My favourite sport is {v}.", "I have a new favourite sport now: {v}.", "What is my favourite sport?",
              ["Which sport do I like most?", "What sport is my favourite?", "Remind me which sport I love most."],
              ["cricket", "rugby", "tennis", "badminton", "swimming", "football", "volleyball", "hockey"]),
    "car": ("I drive a {v}.", "I sold my car. I now drive a {v}.", "What car do I drive?",
            ["Which car do I own?", "What vehicle do I drive?", "Remind me what car I have."],
            ["Toyota", "Honda", "Suzuki", "Nissan", "Mazda", "Kia", "Hyundai", "BMW"]),
    "food": ("My favourite food is {v}.", "My favourite food has changed. It is now {v}.", "What is my favourite food?",
             ["Which food do I like most?", "What dish is my favourite?", "Remind me of my favourite food."],
             ["biryani", "kottu", "hoppers", "pizza", "sushi", "lamprais", "noodles", "dosa"]),
    "bank": ("I bank with {v}.", "I switched banks. I now bank with {v}.", "Which bank do I use?",
             ["Where do I keep my bank account?", "What bank am I with?", "Remind me which bank I use."],
             ["Sampath", "Commercial", "HNB", "Seylan", "NSB", "Peoples", "BOC", "NDB"]),
    "gym": ("I train at {v} gym.", "I joined a different gym. I now train at {v} gym.", "Which gym do I go to?",
            ["Where do I work out?", "What gym am I a member of?", "Remind me which gym I use."],
            ["Atlas", "Titan", "Pulse", "Core", "Summit", "Vertex", "Iron", "Apex"]),
    "colour": ("My favourite colour is {v}.", "My favourite colour is now {v}.", "What is my favourite colour?",
               ["Which colour do I like most?", "What colour is my favourite?", "Remind me of my favourite colour."],
               ["blue", "green", "red", "purple", "orange", "yellow", "black", "white"]),
    "instrument": ("I play the {v}.", "I gave that up. I now play the {v}.", "Which instrument do I play?",
                   ["What instrument do I play?", "Remind me which instrument I play.", "Which musical instrument is mine?"],
                   ["guitar", "piano", "violin", "drums", "flute", "cello", "sitar", "trumpet"]),
    "school": ("My son goes to {v} school.", "My son changed schools. He now goes to {v} school.", "Which school does my son attend?",
               ["Where does my son study?", "What school is my son at?", "Remind me of my son's school."],
               ["Royal", "Trinity", "Ananda", "Nalanda", "Isipathana", "Dharmaraja", "Richmond", "Mahinda"]),
    "language": ("I am learning {v}.", "I switched courses. I am now learning {v}.", "Which language am I learning?",
                 ["What language am I studying?", "Which language course am I taking?", "Remind me what language I am learning."],
                 ["French", "Japanese", "German", "Korean", "Spanish", "Hindi", "Arabic", "Italian"]),
}


def make_persona(rng):
    slots = list(SLOTS)
    rng.shuffle(slots)
    roles = ["superseded"] * 4 + ["erase"] * 2 + ["control"] * 2 + ["unaffected"] * 4
    facts, sessions = [], []
    t = 0
    for slot, role in zip(slots, roles):
        stmt, upd, q, _, pool = SLOTS[slot]
        vals = rng.sample(pool, 2)
        f = {"slot": slot, "role": role, "o": vals[0], "n": vals[1] if role == "superseded" else None}
        facts.append(f)
        reps = 2 if role == "control" else 1
        for _ in range(reps):
            sessions.append({"id": len(sessions), "t": t, "slot": slot, "value": vals[0], "text": stmt.format(v=vals[0])}); t += 1
        if role == "superseded":
            for _ in range(2):  # update dose r = 2
                sessions.append({"id": len(sessions), "t": t + 100, "slot": slot, "value": vals[1], "text": upd.format(v=vals[1])}); t += 1
    deleted = {s["id"] for s in sessions
               if any(f["slot"] == s["slot"] and ((f["role"] == "superseded" and s["value"] == f["n"]) or f["role"] == "erase") for f in facts)}
    return facts, sessions, deleted


def session_examples(s):
    """PLUM-style augmentation: the statement turn plus two question-answer pairs."""
    q = SLOTS[s["slot"]][2]
    sysm = {"role": "system", "content": SYS}
    return [
        {"messages": [sysm, {"role": "user", "content": s["text"]}, {"role": "assistant", "content": "Noted."}]},
        {"messages": [sysm, {"role": "user", "content": q}, {"role": "assistant", "content": s["value"]}]},
        {"messages": [sysm, {"role": "user", "content": s["text"] + " " + q}, {"role": "assistant", "content": s["value"]}]},
    ]


def write_data(dirpath, sessions):
    dirpath.mkdir(parents=True, exist_ok=True)
    ex = [e for s in sessions for e in session_examples(s)]
    with open(dirpath / "train.jsonl", "w") as fh:
        for e in ex:
            fh.write(json.dumps(e) + "\n")
    with open(dirpath / "valid.jsonl", "w") as fh:
        for e in ex[:4]:
            fh.write(json.dumps(e) + "\n")
    return len(ex)


EPOCHS, BATCH, LR_TRAIN = 12, 4, 1e-4


def train_adapter(data_dir, out_dir, n_examples, seed, log):
    iters = max(50, math.ceil(EPOCHS * n_examples / BATCH))
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", BASE, "--train", "--data", str(data_dir),
           "--iters", str(iters), "--batch-size", str(BATCH), "--learning-rate", str(LR_TRAIN),
           "--num-layers", "16", "--adapter-path", str(out_dir), "--mask-prompt", "--seed", str(seed),
           "--steps-per-report", "100", "--steps-per-eval", "100000", "--val-batches", "1", "--save-every", "100000"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout[-2000:] + r.stderr[-3000:])
        raise RuntimeError("lora training failed")
    return time.time() - t0, iters


# ---------------------------------------------------------------- scoring
def ids_for(tok, question, answer):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p = tok.encode(prompt, add_special_tokens=False)
    a = tok.encode(answer, add_special_tokens=False)
    return p, a


def seq_logprob(model, p, a):
    ids = mx.array(p + a)[None]
    logits = model(ids)[:, :-1, :].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = ids[:, 1:]
    tok_lp = mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)[0]
    return tok_lp[len(p) - 1:]


def ell(model, tok, slot, answer):
    """Eq. (1): mean over held-out paraphrases of the length-normalized log-probability."""
    vals = []
    for q in SLOTS[slot][3]:
        p, a = ids_for(tok, q, answer)
        vals.append(seq_logprob(model, p, a).mean())
    return float(mx.mean(mx.stack(vals)).item())


def evaluate(model, tok, facts):
    out = []
    for f in facts:
        r = {"slot": f["slot"], "role": f["role"], "lo": ell(model, tok, f["slot"], f["o"])}
        if f["n"]:
            r["ln"] = ell(model, tok, f["slot"], f["n"])
        pool = [v for v in SLOTS[f["slot"]][4] if v not in (f["o"], f["n"])][:3]
        r["acc"] = int(all(r["lo"] > ell(model, tok, f["slot"], d) for d in pool))
        out.append(r)
    return out


# ---------------------------------------------------------------- unlearning
T_STEPS, LR_UNL, LAMBDA, ALPHA, NPO_BETA, SIMNPO_BETA, FB, RB = 40, 5e-5, 1.0, 4.0, 0.1, 2.5, 8, 8


def qa_items(tok, sessions):
    items = []
    for s in sessions:
        p, a = ids_for(tok, SLOTS[s["slot"]][2], s["value"])
        items.append((p, a, s["id"]))
    return items


def unlearn(model, tok, method, forget, retain, sup_ids, rng):
    model.freeze()
    for _, m in model.named_modules():
        if hasattr(m, "lora_a"):
            m.unfreeze(keys=["lora_a", "lora_b"], strict=False)
    ref = {i: float(seq_logprob(model, p, a).sum().item()) for i, (p, a, _) in enumerate(forget)}
    w = [1.0 + (ALPHA if (method == "STR" and sid in sup_ids) else 0.0) for _, _, sid in retain]

    def loss_fn(fb, rb):
        lf = []
        for i in fb:
            p, a, _ = forget[i]
            lps = seq_logprob(model, p, a)
            if method == "GradDiff":
                lf.append(lps.mean())
            elif method in ("NPO", "STR"):
                x = -NPO_BETA * (lps.sum() - ref[i])
                lf.append(-(2.0 / NPO_BETA) * -mx.logaddexp(mx.array(0.0), -x))
            elif method == "SimNPO":
                x = -SIMNPO_BETA * lps.mean()
                lf.append(-(2.0 / SIMNPO_BETA) * -mx.logaddexp(mx.array(0.0), -x))
        lr_ = [-seq_logprob(model, retain[j][0], retain[j][1]).mean() for j in rb]
        return mx.mean(mx.stack(lf)) + LAMBDA * mx.mean(mx.stack(lr_))

    opt = optim.Adam(learning_rate=LR_UNL)
    vg = nn.value_and_grad(model, loss_fn)
    for _ in range(T_STEPS):
        fb = rng.sample(range(len(forget)), min(FB, len(forget)))
        rb = rng.choices(range(len(retain)), weights=w, k=RB)
        loss, grads = vg(fb, rb)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    return model


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    methods = ["NoUpdate", "GradDiff", "NPO", "SimNPO", "STR"]
    for pi in range(args.personas):
        res_path = out / f"persona_{pi}.json"
        if res_path.exists():
            log(f"persona {pi}: already done"); continue
        rng = random.Random(1000 * args.seed + pi)
        facts, sessions, deleted = make_persona(rng)
        retained = [s for s in sessions if s["id"] not in deleted]
        pdir = out / f"p{pi}"
        n_full = write_data(pdir / "data_full", sessions)
        n_ret = write_data(pdir / "data_retain", retained)
        tt, it = train_adapter(pdir / "data_full", pdir / "theta", n_full, args.seed, log)
        log(f"persona {pi}: theta trained in {tt:.0f}s ({it} iters, {n_full} examples)")
        tr, itr = train_adapter(pdir / "data_retain", pdir / "retrain", n_ret, args.seed, log)
        log(f"persona {pi}: retrain done in {tr:.0f}s ({itr} iters)")

        model, tok = load(BASE, adapter_path=str(pdir / "theta"))
        ev = {"theta": evaluate(model, tok, facts)}
        model, tok = load(BASE, adapter_path=str(pdir / "retrain"))
        ev["retrain"] = evaluate(model, tok, facts)
        forget_s = [s for s in sessions if s["id"] in deleted]
        sup_ids = {s["id"] for s in retained
                   if any(f["role"] == "superseded" and f["slot"] == s["slot"] and s["value"] == f["o"] for f in facts)}
        times = {"theta_train_s": tt, "retrain_s": tr}
        for m in methods:
            if m == "NoUpdate":
                ev[m] = ev["theta"]; times[m] = 0.0; continue
            model, tok = load(BASE, adapter_path=str(pdir / "theta"))
            t0 = time.time()
            unlearn(model, tok, m, qa_items(tok, forget_s), qa_items(tok, retained), sup_ids, random.Random(pi * 7 + 1))
            times[m] = time.time() - t0
            ev[m] = evaluate(model, tok, facts)
            log(f"persona {pi}: {m} done in {times[m]:.0f}s")
        json.dump({"persona": pi, "seed": args.seed, "facts": facts, "n_sessions": len(sessions),
                   "n_deleted": len(deleted), "eval": ev, "times": times,
                   "config": {"epochs": EPOCHS, "batch": BATCH, "lr_train": LR_TRAIN, "lora_layers": 16,
                              "T": T_STEPS, "lr_unlearn": LR_UNL, "lambda": LAMBDA, "alpha": ALPHA,
                              "npo_beta": NPO_BETA, "simnpo_beta": SIMNPO_BETA, "forget_batch": FB, "retain_batch": RB}},
                  open(res_path, "w"), indent=1)
        log(f"persona {pi}: saved")
    log("all done")


if __name__ == "__main__":
    main()
