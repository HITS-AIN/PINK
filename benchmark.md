# Benchmark

## Demo Shapes

Data dim: (4000, 64, 64)
SOM dim: (8, 8)
Rotations: 360
GPU: RTX 2080

| Execution                                           | s/it |
|:---------                                           |  ---:|
| binary with CUDA [^1]                               |    7 |
| binary without CUDA [^2]                            |  135 |
| colab demo without CUDA                             |  127 |
| colab demo without CUDA @ colab.research.google.com |  170 |


[^1]
```
./build/bin/Pink --train /data/pink/shapes_v2.bin som.bin --som-width 8 --som-height 8
```

[^2] `--cuda-off`


## Radio Galaxy Zoo

Data (176750, 124, 124)
SOM (21, 21)

The input data for the SOM training are radio-synthesis images of Radio Galaxy Zoo containing 176750 images of the dimension 124x124.
The SOM layout is hexagonal of the dimension 21x21 which has 331 neurons (see image above). The size of the neurons is 64x64.
The accuracy for the rotational invariance is 1 degree and the flip invariance is used.

|                                   | PINK 1 | PINK 2 |
| :---                              |   ---: |   ---: |
| CPU-1                             |        |  35373 |
| CPU-1 +    NVIDIA Tesla P40       |   3069 |    909 |
| CPU-1 + 2x NVIDIA Tesla P40       |   2069 |    636 |
| CPU-1 + 4x NVIDIA Tesla P40       |   1891 |    858 |
| CPU-2 +    NVIDIA RTX 2080        |        |    673 |
| CPU-3 +    NVIDIA GTX 750 Ti      |        |   7185 |
| CPU-4 + 2x NVIDIA RTX 2080 SUPER  |        |    477 |

All times are in seconds.

  - CPU-1: Intel Gold 5118 (2 sockets, 12 physical cores per socket)
  - CPU-2: Intel Core i7-8700K (1 socket, 6 physical cores per socket)
  - CPU-3: Intel Core i7-4790K (1 socket, 4 physical cores per socket)
  - CPU-4: Intel Gold 6230 (1 socket, 20 physical cores per socket)
