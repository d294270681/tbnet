#!/bin/bash
if [ -d out ]; then
    rm -rf out
fi

mkdir out
cd out || exit

if [ -f "Makefile" ]; then
  make clean
fi

cmake .. \
    -DMINDSPORE_PATH="`pip3.7 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
make
