set -e
./configure --tests --build-gtest --gpu-examples
make -j4
make test
