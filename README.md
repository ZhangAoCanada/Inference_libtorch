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
| libtorch(C++)      | 0.175677                     |
| pytorch(Python)    | 0.216339                     |

*NOTE*: The inference time contains everything, including networks forwarding,  prediction decoding and instance drawing.

## TODO

* Add OpenMP (C++) to see if it could go faster.
* Add ncnn (C++) for comparison
