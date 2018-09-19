#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: build.sh <suffix> <build-type>"
    exit 1
fi

CONAN_DIR="conan-$1"
BUILD_DIR="build-$1"
export CONAN_USER_HOME=$PWD/$CONAN_DIR

if [ ! -d "$CONAN_DIR" ]; then
  mkdir -p $CONAN_DIR
  conan remote add braintwister https://api.bintray.com/conan/braintwister/conan
  conan remote add conan-community https://api.bintray.com/conan/conan-community/conan
  conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
fi

rm -fr $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=$2 ..

if [ $1 -eq "doc" ]; then
    make doc
else
    make 2>&1 |tee make.out
fi
