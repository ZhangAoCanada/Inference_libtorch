# Currently working on nanodet with libtorch

## Current State

> The hardware/software configurations

```yaml
CPU:		Intel i7-8550U (8) @ 2.001GHz
RAM:		16 GB
libtorch:	libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
pytorch:	1.8.0+cpu
```

> The video used fro testing

[Justin Bieber - Anyone | A Cappella Cover](https://www.youtube.com/watch?v=1LhxU6eTCWE)

> Performance comparison(s)

| Library            | Inference Time (s)           |
| ------------------ |:----------------------------:|
| libtorch (C++)     | 0.121933                     |
| ncnn (C++)         | **0.058979**                 |
| pytorch (Python)   | 0.198009          |

*NOTE*: The inference time contains everything, including networks forwarding,  prediction decoding and instance drawing.

*ALSO NOTE*: ncnn is highly optimized for CPU (specially ARM) and GPU (vulkan); libtorch is optimized for GPU (cuda). Further tests need to be done in order to test the performance of these libraries on GPU (cuda).

## TODO

Add OpenMP (C++) to see if it could go faster.
