## Mobile*Pet*V4

my pet project -> replicating the MNv4 architecture from scratch in python.
first time touching attention mechanisms, a good new skill.

the rule I set myself: torch provides tensors, `unfold`/`fold`, and the autograd
tape. everything else is written here. no `nn.Conv2d`, no `nn.BatchNorm2d`, no
`F.cross_entropy`, no `torch.optim`. every backward pass in this repo is one I
derived and typed out.

### layout

blocks live in `Blocks/<Name>/` and always follow the same shape:

- `block.py` -> a `torch.autograd.Function` with hand-written `forward` and
  `backward`. stateless, pure math.
- `module.py` -> the `nn.Module` that owns the `nn.Parameter`s and calls
  `Block.apply(...)`. only exists where there is something to own.

that split turned out to matter more than I expected. it keeps the gradient math
testable on its own, and it makes the parameter/buffer distinction obvious
(BatchNorm's `running_mean` is a buffer, its `gamma` is a parameter, and the
optimizer only ever sees the second one).

```
Blocks/          Conv2D, DepthwiseConv, PointwiseConv, BatchNorm, ReLU,
                 ConvBNAct, UniversalInvertedBottleneck, FusedInvertedBottleneck,
                 SqueezeExcite, LayerScale, Dropout, GlobalAveragePooling,
                 Linear, FusedCrossEntropy, MultiQueryAttention, VanillaAttention
model/mnv4.py    the config table + the builder that turns it into modules
training/        optimizer (AdamW), param grouping, train/eval loop, mock data
scripts/         the demo runner and the chart generator
```

### the model

`model/mnv4.py` keeps the architecture as plain data and builds it in a loop.
MNv4 ships as five variants that differ only in numbers, so hand-writing each one
would have been five chances to typo a channel count.

block specs use timm's notation, which is compact and worth learning: `a` is the
first depthwise kernel, `k` the second, `0` means that conv is absent, `e` is the
expansion ratio relative to the input, `c` is output channels.

MNv4-Conv-Small at 1000 classes comes out at **3,772,744 params**. the paper says
~3.8M, so the table is right.

### the optimizer

this is the part I spent the most time on, and the part I learned the most from.
`training/optimizer.py` is AdamW written from the paper: EMAs of the gradient and
the squared gradient, bias correction, decoupled weight decay, parameter groups.

checked against `torch.optim.AdamW` over 200 steps, same init, same gradients:

| weight_decay | float32 | float64 |
| --- | --- | --- |
| 0.0 | 2.4e-07 | 4.4e-16 |
| 0.01 | 8.8e-06 | 7.1e-15 |
| 0.1 | 5.7e-06 | 8.9e-16 |

float64 lands on machine epsilon, so the float32 column is rounding, not error.
it is the same algorithm.

three bugs I hit on the way, all of which cost me real time:

1. **storing the bias-corrected value back into the state.** `m` and `mhat` are
   two different things. the correction factor `1/(1-beta^t)` is different every
   step, so if you bake step 1's correction into the slot, step 2 corrects the
   already-corrected value and they compound. `v` blew up to 8.35e9 by step 5 and
   the effective step size collapsed to nothing. nothing crashed. the loss just
   sat there.
2. **applying the decay after the adam step instead of before.** the paper uses
   the pre-update weight in the decay term. doing it after multiplies the adam
   step by `(1 - lr*wd)` as well. tiny in absolute terms, but it is the difference
   between matching torch and not.
3. **a `setdefault` loop that iterated the dict it was writing into.** it can
   never add a missing key that way. param groups silently came out without
   `betas` and blew up at step time.

`training/sort_params.py` splits parameters into a decayed group and a
non-decayed one on `p.dim() >= 2`. conv kernels and weight matrices are 2D+,
biases and BN gamma/beta are 1D, so the rule separates them cleanly with no naming
conventions to maintain. decaying a BN gamma is pointless anyway, the next
normalization undoes it.

### running it

```
python scripts/train_demo.py
python scripts/plot_run.py
```

the demo trains on generated images rather than a real dataset: six shape classes
(horizontal bar, vertical bar, diagonal, disc, square, cross) with randomised
position, size and both colours, plus gaussian noise. not hard, but it has real
structure, so the curve means something. torch here is CPU-only and torchvision is
not installed, so this keeps the whole thing dependency-free and under three
minutes.

### the run

MNv4-Conv-Small, 64px, 1520 train / 380 val, 6 classes, batch 32, AdamW at
lr=2e-3 wd=0.05. 12 epochs, 2.7 minutes on CPU.

![loss](runs/demo/loss.svg)

![accuracy](runs/demo/accuracy.svg)

15.3% at init (chance is 16.7%) up to ~99%. epoch 1 is worse than random on val
before it catches, which I assume is the BatchNorm running stats still being
garbage that early.

then I wanted to see whether my decoupled decay was actually doing anything. in a
single run you cannot tell, the gradient updates are far bigger than
`lr*wd = 1e-4` per step and the weight norm looks flat. so: same seed, same data,
same everything, only `weight_decay` differs.

```
python scripts/train_demo.py --wd 0.0 --out runs/nodecay
```

![weight decay ablation](runs/demo/weight_rms.svg)

the gap opens up monotonically and ends at 4.2%. that is the decay working.

charts are hand-written SVG (`scripts/plot_run.py`) so the repo does not pull in
matplotlib for three line plots.

### not done yet

- **MQA is not wired into the model.** the block and module exist and the math is
  there, but `MobileMQA.__init__` takes raw weight shapes instead of channel
  counts, and `kv_stride` is hardcoded to 2 in `forward`. the hybrid variants need
  stride 1 for half their attention blocks, so it needs a refactor before it can
  be driven from a config table.
- `FusedInvertedBottleneck.forward` references `self.b_project`, which `__init__`
  never defines. it raises the moment it is called. UIB made it mostly redundant
  so I have not decided whether to fix or delete it.
- `tests/` is empty. the optimizer checks above were all run ad hoc and should be
  pinned down properly, especially the torch parity one.
- no LR schedule. warmup + cosine belongs in `training/`, not in the optimizer.
- no `state_dict` / `load_state_dict` on the optimizer, so a run cannot be resumed
  from a checkpoint. restoring only the weights and throwing away `m` and `v`
  would spike the loss.
- never trained on real images.

### notes to self

- `p -= ...` on a leaf tensor with `requires_grad=True` raises. the optimizer step
  has to happen under `torch.no_grad()`, otherwise you are recording parameter
  updates into the graph.
- `p = p - lr * g` rebinds a local name and does nothing. it has to be in-place.
- the first Adam step is exactly `-lr * sign(g)` regardless of how big the
  gradient is, because with one observation the corrections cancel the mixing
  factors exactly. best single unit test for bias correction.
- `FusedCrossEntropy` returns one loss per sample, not a scalar. `.mean()` before
  `.backward()`.
- a test where the expected-unchanged value happens to be 0 proves nothing. BN
  `beta` and `Linear.b` both init to zeros, and decay is multiplicative, so they
  look "correctly skipped" even when they are not.
