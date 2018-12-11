#!/bin/bash

find .. -type f \( \
    -name "*.cpp" -o \
    -name "*.h" -o \
    -name "*.cu" -o \
    -name "*.py" -o \
    -name "*.txt" \) \
    -exec bash -c 'expand -i -t 4 "$0" | sponge "$0"' {} \;
