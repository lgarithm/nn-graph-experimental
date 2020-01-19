#!/bin/sh
set -e

./configure --enable-cuda --gpu-examples
# make 2>err.log | tee out.log
make -j $(nproc)
