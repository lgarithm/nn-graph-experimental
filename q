#!/bin/sh
set -e

./configure --enable-cuda
make 2>err.log | tee out.log
# make -j 8
