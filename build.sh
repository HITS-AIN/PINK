#!/bin/bash

SUFFIX=""
if [ $# -eq 1 ]; then
    SUFFIX="-$1"
fi

CONAN_DIR="conan$SUFFIX"
BUILD_DIR="build$SUFFIX"
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

cmake -DCMAKE_BUILD_TYPE=Release ..
make 2>&1 |tee make.out