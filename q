set -e
./configure --tests --build-gtest
make -j4
make test
