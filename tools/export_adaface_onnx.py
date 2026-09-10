"""Export AdaFace IR-101 (CVLFace official WebFace4M weights) to ONNX and verify.

Weights source (MIT, official author):
  https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m
  - pretrained_model/model.pt  (state_dict)
  - models/iresnet/model.py    (architecture)

Download the repo files locally, then:
  python tools/export_adaface_onnx.py --repo-dir <local repo dir> --out models/adaface_ir101_webface4m.onnx
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True, help="dir containing models/ and pretrained_model/")
    parser.add_argument("--out", required=True, help="output onnx path")
    parser.add_argument("--cross-check", default=None, help="optional community onnx for cross check")
    args = parser.parse_args()

    repo = Path(args.repo_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch

    # model.py imports fvcore only for its __main__ block; stub it out.
    sys.modules.setdefault("fvcore", types.ModuleType("fvcore"))
    sys.modules.setdefault("fvcore.nn", types.ModuleType("fvcore.nn"))
    sys.modules["fvcore.nn"].flop_count = lambda *a, **k: ({}, None)

    # Load models/iresnet/model.py directly (bypass package __init__ which pulls
    # in safetensors/omegaconf that we don't need).
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cvlface_iresnet_model", repo / "models" / "iresnet" / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    IR_101 = module.IR_101

    state_dict = torch.load(repo / "pretrained_model" / "model.pt", map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if all(k.startswith("net.") for k in state_dict):
        state_dict = {k[len("net."):]: v for k, v in state_dict.items()}
    elif any(k.startswith("net.") for k in state_dict):
        raise RuntimeError(f"unexpected mixed key prefixes: {list(state_dict)[:5]}")

    model = IR_101(input_size=(112, 112))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"loaded weights: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )
    print(f"exported: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # ---- verification: torch vs onnxruntime ----
    import onnxruntime as ort

    rng = np.random.RandomState(42)
    # Mimic real preprocessing range: (x/255 - 0.5)/0.5 in [-1, 1]
    batch = rng.uniform(-1.0, 1.0, size=(16, 3, 112, 112)).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).numpy()

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: batch})[0]

    def cos(a, b):
        a = a.reshape(len(a), -1)
        b = b.reshape(len(b), -1)
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return num / den

    c_torch_ort = cos(torch_out, ort_out)
    print(f"cos(torch, exported onnx): min={c_torch_ort.min():.6f} mean={c_torch_ort.mean():.6f}")

    ok = c_torch_ort.min() >= 0.999
    if args.cross_check:
        sess2 = ort.InferenceSession(args.cross_check, providers=["CPUExecutionProvider"])
        in2 = sess2.get_inputs()[0]
        print(f"cross-check model inputs: {[(i.name, i.shape) for i in sess2.get_inputs()]}")
        out2 = sess2.run(None, {in2.name: batch})[0]
        c_cross = cos(ort_out, out2)
        print(f"cos(exported, community onnx): min={c_cross.min():.6f} mean={c_cross.mean():.6f}")
        # embedding order/flip differences can make cosine negative-equal; check both
        c_cross_flipped = cos(ort_out, -out2)
        if c_cross_flipped.min() > c_cross.min():
            print(f"NOTE: community onnx embedding is sign-flipped (cos min={c_cross_flipped.min():.6f})")

    if ok:
        print("VERIFICATION PASSED")
        return 0
    print("VERIFICATION FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
