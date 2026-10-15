"""MoeVerify / MoeDecode time per op (REP nodes over distinct weights, CUDA events). usage: python tools/bench_moe.py [R] [op]"""
import os, sys, torch, onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_bolt_ops import session, vi, BF, F32, U8, I32, helper
R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
op = sys.argv[2] if len(sys.argv) > 2 else "MoeVerify"
E, I, topk, REP = 304, 2304, 9, 6
torch.manual_seed(0)
inputs, feeds, nodes, outs = [("z", BF, [R, 1024]), ("i", I32, [R, topk]), ("w", F32, [R, topk])], {}, [], []
feeds["z"] = (torch.randn(R, 1024) * 0.5).bfloat16()
feeds["i"] = torch.stack([torch.randperm(E)[:topk] for _ in range(R)]).int()
feeds["w"] = torch.rand(R, topk)
for r in range(REP):
    for n, t in ((f"w1_{r}", torch.randint(0, 256, (E, I, 512), dtype=torch.uint8)), (f"s1_{r}", (torch.rand(E, I, 32) * 0.003).bfloat16()),
                 (f"w2_{r}", torch.randint(0, 256, (E, 1024, I // 2), dtype=torch.uint8)), (f"s2_{r}", (torch.rand(E, 1024, I // 32) * 0.003).bfloat16())):
        inputs.append((n, {torch.uint8: U8, torch.bfloat16: BF}[t.dtype], list(t.shape))); feeds[n] = t
    o = [f"o{r}_{k}" for k in range(4 if op == "MoeVerify" else 3)]
    nodes.append(helper.make_node(op, ["z", "i", "w", f"w1_{r}", f"s1_{r}", f"w2_{r}", f"s2_{r}"], o, domain="bolt", topk=topk)); outs += o
otypes = [BF, BF, F32, I32][:len(outs) // REP] * REP
g = helper.make_graph(nodes, "b", [vi(n, t, s) for n, t, s in inputs], [vi(o, t, None) for o, t in zip(outs, otypes)])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("bolt", 1)]); m.ir_version = 10
sess = session(m); b = sess.io_binding(); keep = []
for n, t in feeds.items():
    ov = ort.OrtValue.from_dlpack(t.cuda().contiguous().__dlpack__()); keep.append(ov); b.bind_ortvalue_input(n, ov)
for o in outs: b.bind_output(o, "cuda", 0)
for _ in range(3): sess.run_with_iobinding(b)
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(5): sess.run_with_iobinding(b)
e1.record(); torch.cuda.synchronize()
print(f"{op} R={R} E={E} I={I}: {e0.elapsed_time(e1) / 5 / REP * 1e3:.1f} us/op")
