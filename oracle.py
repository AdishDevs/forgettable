"""Oracle diagnostic: KL matching of the unlearned adapter to the retrained reference R_u.

Starts from theta (as every method does), runs T steps of Adam on
  KL(p_R || p_phi) over answer positions of forget and retain training-format items,
with the same T, learning rate and batch sizes as pilot.py. Needs R_u, so it is a
diagnostic upper bound, not a method. Writes persona_{pi}_oracle.json next to the results.
Usage: .venv/bin/python oracle.py results 0 7 0
"""
import json, random, sys, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
import pilot


def answer_logprobs(model, p, a):
    ids = mx.array(p + a)[None]
    logits = model(ids)[:, len(p) - 1:-1, :].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def run(rdir, pi, seed):
    out = rdir / f"persona_{pi}_oracle.json"
    if out.exists():
        return
    facts, sessions, deleted = pilot.make_persona(random.Random(pi))
    d = json.load(open(rdir / f"persona_{pi}.json"))
    assert [f["o"] for f in facts] == [f["o"] for f in d["facts"]], "persona mismatch"
    retained = [s for s in sessions if s["id"] not in deleted]
    forget_s = [s for s in sessions if s["id"] in deleted]
    model, tok = load(pilot.BASE, adapter_path=str(rdir / f"p{pi}" / "theta"))
    teacher, _ = load(pilot.BASE, adapter_path=str(rdir / f"p{pi}" / "retrain"))
    forget, retain = pilot.qa_items(tok, forget_s), pilot.qa_items(tok, retained)
    items = forget + retain
    tgt = [mx.stop_gradient(answer_logprobs(teacher, p, a)) for p, a, _ in items]
    mx.eval(tgt)
    del teacher
    model.freeze()
    for _, m in model.named_modules():
        if hasattr(m, "lora_a"):
            m.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    def loss_fn(batch):
        kls = []
        for i in batch:
            p, a, _ = items[i]
            lq = answer_logprobs(model, p, a)
            kls.append((mx.exp(tgt[i]) * (tgt[i] - lq)).sum(-1).mean())
        return mx.mean(mx.stack(kls))

    rng = random.Random(pi * 7 + 1 + 1000 * seed)
    opt = optim.Adam(learning_rate=pilot.LR_UNL)
    vg = nn.value_and_grad(model, loss_fn)
    t0 = time.time()
    for _ in range(pilot.T_STEPS):
        fb = rng.sample(range(len(forget)), min(pilot.FB, len(forget)))
        rb = [len(forget) + j for j in rng.choices(range(len(retain)), k=pilot.RB)]
        loss, grads = vg(fb + rb)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    t = time.time() - t0
    ev = pilot.evaluate(model, tok, facts)
    json.dump({"persona": pi, "seed": seed, "time_s": t, "Oracle": ev}, open(out, "w"), indent=1)
    print(f"{rdir.name} persona {pi}: oracle {t:.0f}s", flush=True)


if __name__ == "__main__":
    rdir = pilot.HERE / sys.argv[1]
    for pi in range(int(sys.argv[2]), int(sys.argv[3])):
        run(rdir, pi, int(sys.argv[4]))
