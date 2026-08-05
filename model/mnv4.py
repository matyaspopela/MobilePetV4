import torch
import torch.nn as nn

from Blocks.ConvBNAct.block import ConvBNAct
from Blocks.UniversalInvertedBottleneck.block import UniversalInvertedBottleneck
from Blocks.GlobalAveragePooling.block import GlobalAveragePooling
from Blocks.Linear.module import Linear


def build_block(spec, c_in):

    kind = spec["kind"]

    if kind == "conv":
        block = ConvBNAct(c_in, spec["c_out"], "standard",
                          kernel_size=spec.get("k", 3),
                          stride=spec.get("stride", 1))
        return block, spec["c_out"]

    if kind == "pointwise":
        block = ConvBNAct(c_in, spec["c_out"], "pointwise")
        return block, spec["c_out"]

    if kind == "uib":
        # timm notation: a = first depthwise kernel, k = second. 0 means "no such conv".
        a = spec.get("a", 0)
        k = spec.get("k", 0)
        block = UniversalInvertedBottleneck(
            c_in=c_in,
            c_mid=c_in * spec["e"],          # expansion is relative to the input
            c_out=spec["c_out"],
            dw_before=a > 0,
            dw_after=k > 0,
            dw_before_kernel=a if a > 0 else 3,
            dw_after_kernel=k if k > 0 else 3,
            stride=spec.get("stride", 1),
        )
        return block, spec["c_out"]

    raise ValueError(f"unknown block kind: {kind!r}")

class MobileNetV4(nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super().__init__()

        c = 3
        self.stem, c = build_block(cfg["stem"], c)

        blocks = []

        for spec in cfg["blocks"]:
            for i in range(spec.get("repeat", 1)):
                repeated = dict(spec)
                if i > 0:
                    repeated["stride"] = 1   # only the first of a run downsamples
                block, c = build_block(repeated, c)
                blocks.append(block)

        self.blocks = nn.Sequential(*blocks) # registers blocks else they would not appear in model.parameters()

        self.head = ConvBNAct(c, cfg["head_channels"], "pointwise")
        self.fc1 = Linear(cfg["head_channels"], cfg["classifier_hidden"])
        self.fc2 = Linear(cfg["classifier_hidden"], num_classes)

    def forward(self, input):
        out = self.stem(input)
        out = self.blocks(out)
        out = self.head(out)
        out = GlobalAveragePooling.apply(out)
        out = self.fc1(out)
        return self.fc2(out)

# MODEL CONFIGS BELOW

MNV4_CONV_SMALL = {
    "stem": {"kind": "conv", "c_out": 32, "k": 3, "stride": 2},
    "blocks": [
        # stage 0
        {"kind": "conv",      "c_out": 32, "k": 3, "stride": 2},
        {"kind": "pointwise", "c_out": 32},
        # stage 1
        {"kind": "conv",      "c_out": 96, "k": 3, "stride": 2},
        {"kind": "pointwise", "c_out": 64},
        # stage 2
        {"kind": "uib", "c_out": 96,  "e": 3, "stride": 2, "a": 5, "k": 5},
        {"kind": "uib", "c_out": 96,  "e": 2, "stride": 1, "a": 0, "k": 3, "repeat": 4},
        {"kind": "uib", "c_out": 96,  "e": 4, "stride": 1, "a": 3, "k": 0},
        # stage 3
        {"kind": "uib", "c_out": 128, "e": 6, "stride": 2, "a": 3, "k": 3},
        {"kind": "uib", "c_out": 128, "e": 4, "stride": 1, "a": 5, "k": 5},
        {"kind": "uib", "c_out": 128, "e": 4, "stride": 1, "a": 0, "k": 5},
        {"kind": "uib", "c_out": 128, "e": 3, "stride": 1, "a": 0, "k": 5},
        {"kind": "uib", "c_out": 128, "e": 4, "stride": 1, "a": 0, "k": 3, "repeat": 2},
    ],
    # stage 4 (the 1x1 -> 960) is self.head, then GAP, then 1280, then classifier
    "head_channels": 960,
    "classifier_hidden": 1280,
}
