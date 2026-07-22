# MobilePetV4

My manual implementation of the MobileNetV4 Hybrid architecture.

The goal of this project is not necessarily a good benchmark or results, rather learning the architecture in depth, by reimplementing it.

I all of the basic forward operations manually, and also manually derived their backward passes.
I used torch.autograd.Function for this and torch.nn.Functional for optimized unfolding and basic matrix (tensor) operations.

## Where i stopped deriving gradients manually:
Inverted Bottleneck, Fused Inverted Bottleneck, Universal Inverted bottleneck ...

**These were just simply composed from previous blocks like depthwise conv, pointwise conv that i had already written. I found it tedious and not particularly enlighenting, having to rewrite all of that agian into a bigger class, so instead for the things that reused previous blocks largely, i made a nn.module and called the smaller blocks there.*

I then obviously came back to manual deriving when i built the MQA attention block which was probably the most difficult part of this entire thing.

just realised i also accidentaly coded pointwiseconv 2 times. lol