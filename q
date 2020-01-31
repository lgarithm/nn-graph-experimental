#!/bin/sh
set -e

now() { date +%s; }

measure() {
    local begin=$(now)
    $@
    local end=$(now)
    local duration=$((end - begin))
    echo "$@ took $duration"
}

flags() {
    echo --gpu-examples
    echo --enable-cuda
    echo --enable-trace
}

rebuild() {
    ./configure $(flags)
    # make 2>err.log | tee out.log
    make -j $(nproc)
}

measure rebuild
measure ./bin/train-mnist-cnn gpu
