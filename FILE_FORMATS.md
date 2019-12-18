# Binary Data Format

Every file can have multiple readable comment lines at first which all must have the character `#` as first letter.

All indices are decoded as 32-bit integer. The file format version is 2. Currently,
only 32-bit floating point numbers will be supported as data type, but we will be prepared for the future.

  - 0: float 32
  - 1: float 64
  - 2: integer 8
  - 3: integer 16
  - 4: integer 32
  - 5: integer 64
  - 6: unsigned integer 8
  - 7: unsigned integer 16
  - 8: unsigned integer 32
  - 9: unsigned integer 64
  
The layout for data, som, and neuron can be

  - 0: cartesian
  - 1: hexagonal
  
followed by the dimensionality and the dimensions.

## Cartesian layout

The data layout is row-major (C-style), which means that the strides grow from right to left.

### 2-dim example 10x10

The layout description is: `0 2 10 10`, the strides will be `(10, 0)` and the element `[i, j]` is at memory position `i * 10 + j`.

### 3-dim example 10x10x3

The layout description is: `0 3 10 10 3`, the strides will be `(30, 3, 0)` and the element `[i, j, k]` is at memory position `i * 30 + j * 3 + k`.

## Hexagonal layout
### 2-dim example 11x11

The layout description is: `1 2 11 11`. The row- and column-dimension must be odd and equal (`d`). The memory position `p` can be calculated by

```python
r = int((d - 1) / 2)
row_size = [d - abs(r - i) for i in range(d)]
row_offset = [sum(size[0:i]) for i in range(d+1)]
p = offset[i] + j
if (r > i) p -= r - i
```
Please see also https://www.redblobgames.com/grids/hexagons/#map-storage


## Data file for training and mapping

```
<file format version> 0 <data-type> <number of entries> <data layout> <data>
```
  
Example:

A data file containing 1000 entries of a 2-dimensional image with 128x128 pixels looks like

```
2 0 0 1000 0 2 128 128 <16384000 floating point entries>
```

## SOM file

```
<file format version> 1 <data-type> <som layout> <neuron layout> <data>
```

## Mapping file

```
<file format version> 2 <data-type> <number of entries> <som layout> <data>
```

## Best rotation and flipping parameter file

```
<file format version> 3 <number of entries> <som layout> <data>
```
  
The data section contains a bool (is flipped) and a 32-bit float number (angle in radian) for each neuron.
