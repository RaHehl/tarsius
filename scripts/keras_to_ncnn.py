"""
Convert a Keras 3 segmentation finder (.keras) to an ncnn param/bin pair.

Handles exactly the layer set the meter finders use: Input, Rescaling
(folded into the first convolution), Conv2D, DepthwiseConv2D,
BatchNormalization and ReLU (folded into the preceding convolution),
nearest UpSampling2D and Concatenate. Anything else fails loudly.

Reads the archive with h5py and numpy only, so the output depends on the
input bytes and nothing else. Weights are stored as fp16 like ncnnoptimize
does, biases as fp32.

    python scripts/keras_to_ncnn.py finder.keras out/finder
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FP16_TAG = 0x01306B47
SAME_UPPER = -233
# Keras 3 stores weights under per-class counters in layer order, not
# under the layer names: the third Conv2D is "conv2d_2".
STORE_NAMES = {
    "Conv2D": "conv2d",
    "DepthwiseConv2D": "depthwise_conv2d",
    "BatchNormalization": "batch_normalization",
}

HANDLERS = {
    "InputLayer": "input",
    "Rescaling": "rescaling",
    "Conv2D": "conv2d",
    "DepthwiseConv2D": "depthwise_conv2d",
    "BatchNormalization": "batch_normalization",
    "ReLU": "relu",
    "UpSampling2D": "upsampling2d",
    "Concatenate": "concatenate",
}

type Op = dict[str, Any]


def inbound(layer: dict[str, Any]) -> list[str]:
    return [
        item["config"]["keras_history"][0]
        for node in layer.get("inbound_nodes", [])
        for arg in node["args"]
        for item in (arg if isinstance(arg, list) else [arg])
        if isinstance(item, dict) and "keras_history" in item.get("config", {})
    ]


def load(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[np.ndarray]]]:
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        weights_bytes = archive.read("model.weights.h5")
    layers = config["config"]["layers"]
    weights: dict[str, list[np.ndarray]] = {}
    seen: Counter[str] = Counter()
    with h5py.File(BytesIO(weights_bytes), "r") as h5:
        for layer in layers:
            base = STORE_NAMES.get(layer["class_name"])
            if base is None:
                continue
            stored = base if not seen[base] else f"{base}_{seen[base]}"
            seen[base] += 1
            group = h5["layers"][stored]["vars"]
            weights[layer["config"]["name"]] = [
                np.asarray(group[str(i)], np.float32) for i in range(len(group))
            ]
    return layers, weights


class Converter:
    """Walks the Keras graph in order and folds it into ncnn layers."""

    def __init__(self, weights: dict[str, list[np.ndarray]], consumers: Counter[str]):
        self.weights = weights
        self.consumers = consumers
        self.ops: list[Op] = []
        self.alias: dict[str, str] = {}
        self.producer: dict[str, Op] = {}
        self.scale_of: dict[str, float] = {}

    def add(self, kind: str, cfg: dict[str, Any], srcs: list[str]) -> None:
        name = cfg["name"]
        if kind not in HANDLERS:
            raise SystemExit(f"unsupported layer {kind} ({name})")
        handler = getattr(self, f"on_{HANDLERS[kind]}")
        self.alias[name] = handler(name, cfg, srcs)

    def on_input(self, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        _, h, w, c = cfg["batch_shape"]
        self.ops.append(
            {"type": "Input", "name": name, "ins": [], "params": {0: w, 1: h, 2: c}}
        )
        return name

    def on_rescaling(self, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        assert cfg["offset"] == 0.0 and self.consumers[name] == 1
        self.scale_of[srcs[0]] = float(cfg["scale"])
        return srcs[0]

    def on_conv2d(self, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        return self._conv("Convolution", name, cfg, srcs)

    def on_depthwise_conv2d(
        self, name: str, cfg: dict[str, Any], srcs: list[str]
    ) -> str:
        assert cfg["depth_multiplier"] == 1
        return self._conv("ConvolutionDepthWise", name, cfg, srcs)

    def _conv(self, kind: str, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        kh, kw = cfg["kernel_size"]
        assert cfg["padding"] == "same" or (kh, kw) == (1, 1)
        assert cfg["activation"] == "linear"
        assert tuple(cfg.get("dilation_rate", (1, 1))) == (1, 1)
        kernel = self.weights[name][0]
        if srcs[0] in self.scale_of:
            kernel = kernel * self.scale_of.pop(srcs[0])
        if kind == "Convolution":
            kernel = kernel.transpose(3, 2, 0, 1)
        else:
            kernel = kernel[..., 0].transpose(2, 0, 1)
        out_c = kernel.shape[0]
        bias = self.weights[name][1] if cfg["use_bias"] else np.zeros(out_c, np.float32)
        sh, sw = cfg["strides"]
        op = {
            "type": kind,
            "name": name,
            "ins": srcs,
            "kernel": kernel,
            "bias": bias,
            "relu": False,
        }
        op.update(out_c=out_c, kh=kh, kw=kw, sh=sh, sw=sw)
        self.ops.append(op)
        self.producer[name] = op
        return name

    def on_batch_normalization(
        self, name: str, cfg: dict[str, Any], srcs: list[str]
    ) -> str:
        op = self._sole_producer(srcs[0], name)
        gamma, beta, mean, var = self.weights[name]
        scale = gamma / np.sqrt(var + cfg["epsilon"])
        op["kernel"] = op["kernel"] * scale.reshape(
            (-1,) + (1,) * (op["kernel"].ndim - 1)
        )
        op["bias"] = (op["bias"] - mean) * scale + beta
        return srcs[0]

    def on_relu(self, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        assert not cfg.get("max_value") and not cfg.get("negative_slope")
        assert not cfg.get("threshold")
        self._sole_producer(srcs[0], name)["relu"] = True
        return srcs[0]

    def _sole_producer(self, src: str, name: str) -> Op:
        """Return the convolution to fold into; it must feed only ``name``."""
        op = self.producer[src]
        assert self.consumers[op["name"]] == 1 and not op["relu"]
        # Whatever reads this layer now reads the folded convolution.
        self.consumers[op["name"]] = self.consumers[name]
        return op

    def on_upsampling2d(self, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        assert cfg["interpolation"] == "nearest"
        sh, sw = cfg["size"]
        params = {0: 1, 1: float(sh), 2: float(sw)}
        self.ops.append({"type": "Interp", "name": name, "ins": srcs, "params": params})
        return name

    def on_concatenate(self, name: str, cfg: dict[str, Any], srcs: list[str]) -> str:
        assert cfg["axis"] == -1
        self.ops.append({"type": "Concat", "name": name, "ins": srcs, "params": {0: 0}})
        return name


def convert(path: Path) -> tuple[str, bytes, dict[str, Any]]:
    layers, weights = load(path)
    converter = Converter(
        weights, Counter(s for layer in layers for s in inbound(layer))
    )
    for layer in layers:
        srcs = [converter.alias[s] for s in inbound(layer)]
        converter.add(layer["class_name"], layer["config"], srcs)
    assert not converter.scale_of, "a Rescaling layer was not followed by a convolution"
    ops = converter.ops
    info = {
        "input": [ops[0]["params"][1], ops[0]["params"][0]],
        "output": ops[-1]["name"],
    }
    return emit(ops), weights_blob(ops), info


def conv_params(op: Op) -> dict[int, int]:
    params = {0: op["out_c"], 1: op["kw"], 11: op["kh"], 3: op["sw"], 13: op["sh"]}
    if op["kh"] > 1 or op["kw"] > 1:
        params[4] = SAME_UPPER
    params[5] = 1
    params[6] = int(op["kernel"].size)
    if op["type"] == "ConvolutionDepthWise":
        params[7] = op["out_c"]
    if op["relu"]:
        params[9] = 1
    return params


def emit(ops: list[Op]) -> str:
    # ncnn needs a Split layer for every blob read by more than one layer.
    uses = Counter(src for op in ops for src in op["ins"])
    taken: Counter[str] = Counter()
    lines: list[str] = []
    blobs = 0
    for op in ops:
        ins = []
        for src in op["ins"]:
            if uses[src] > 1:
                ins.append(f"{src}_split{taken[src]}")
                taken[src] += 1
            else:
                ins.append(src)
        params = conv_params(op) if "kernel" in op else op["params"]
        text = " ".join(
            f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
        )
        names = " ".join([*ins, op["name"]])
        lines.append(f"{op['type']:<24}{op['name']:<24}{len(ins)} 1 {names} {text}")
        blobs += 1
        if (count := uses[op["name"]]) > 1:
            outs = " ".join(f"{op['name']}_split{i}" for i in range(count))
            split = f"{op['name']}_split"
            lines.append(f"{'Split':<24}{split:<24}1 {count} {op['name']} {outs}")
            blobs += count
    return f"7767517\n{len(lines)} {blobs}\n" + "\n".join(lines) + "\n"


def weights_blob(ops: list[Op]) -> bytes:
    out = bytearray()
    for op in ops:
        if "kernel" not in op:
            continue
        half = np.ascontiguousarray(op["kernel"], np.float32).astype("<f2").tobytes()
        out += np.uint32(FP16_TAG).astype("<u4").tobytes() + half
        out += b"\0" * (-len(half) % 4)
        out += np.ascontiguousarray(op["bias"], "<f4").tobytes()
    return bytes(out)


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    param, weights, info = convert(source)
    target.with_suffix(".param").write_text(param)
    target.with_suffix(".bin").write_bytes(weights)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
