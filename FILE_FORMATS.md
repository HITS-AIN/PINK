# Binary Data Format

Arrays are in FORTRAN notation: First index is the slowest, last index is the fastest.

## Image file

    (1)(integer) number_of_images
    (1)(integer) number_of_channels
    (1)(integer) width
    (1)(integer) height
    (number_of_images, number_of_channels, width, height)(float) pixels

## SOM file

    (1)(integer) number_of_channels
    (1)(integer) SOM_width
    (1)(integer) SOM_height
    (1)(integer) SOM_depth
    (1)(integer) neuron_width
    (1)(integer) neuron_height
    (SOM_width, SOM_height, SOM_depth, number_of_channels, neuron_width, neuron_height)(float) pixels

## Mapping file

    (1)(integer) number_of_images
    (1)(integer) SOM_width
    (1)(integer) SOM_height
    (1)(integer) SOM_depth
    (number_of_images, SOM_width, SOM_height, SOM_depth)(float) euclidian_distance

## Best rotation and flipping parameter file

    (1)(integer) number_of_images
    (1)(integer) SOM_width
    (1)(integer) SOM_height
    (1)(integer) SOM_depth
    (number_of_images, SOM_width, SOM_height, SOM_depth)(bool) flipped, angle_in_radian
