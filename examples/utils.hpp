#pragma once
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

template <typename T1, typename T2>
std::vector<T1> firsts(const std::vector<std::pair<T1, T2>> &pairs)
{
    std::vector<T1> v(pairs.size());
    std::transform(pairs.begin(), pairs.end(), v.begin(),
                   [](auto p) { return p.first; });
    return v;
}

void show_accuracy(float acc, int epoch, int step)
{
    const int len = 50;
    const int bar = static_cast<int>(len * acc);
    std::cerr << std::right << "epoch " << std::setw(4) << epoch << ", "
              << "step " << std::setw(4) << step << ", "
              << "accuracy: " << std::setw(10) << acc << " | " << std::setw(len)
              << std::left << std::string(bar, '#') << " | " << std::endl;
}
