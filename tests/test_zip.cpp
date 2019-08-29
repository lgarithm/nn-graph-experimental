#include "testing.hpp"

#include <experimental/zip>
#include <ttl/range>
#include <vector>

using std::experimental::zip;
using ttl::range;

/*
TEST(zip_test, test1)
{
    const int n = 10;
    int s = 0;
    for (auto [i, j] : zip(range(n), range(n))) { s += i + j; }
    ASSERT_EQ(s, 90);
}
*/

/*
TEST(zip_test, test2)
{
    const int n = 10;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    int s = 0;
    for (auto [i, j] : zip(range(n), v)) { s += i + j; }
    ASSERT_EQ(s, 90);
}
*/

TEST(zip_test, test3)
{
    const int n = 10;
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    int s = 0;
    for (auto [i, j] : zip(v, v)) { s += i + j; }
    ASSERT_EQ(s, 90);
}
