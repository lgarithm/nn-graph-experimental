#pragma once
#include <set>

namespace nn::graph::internal
{
class node;

// execution keeps the state within one run
class execution
{
    using key_t = const node *;

    std::set<key_t> done_;

  public:
    void done(key_t key)
    {
        // FIXME: check dup
        done_.insert(key);
    }

    bool is_done(key_t key) const { return done_.count(key) > 0; }
};
}  // namespace nn::graph::internal
