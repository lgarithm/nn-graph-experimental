#pragma once
#include <fstream>
#include <stdml/filesystem>
#include <string>

#include <ttl/nn/bits/ops/io.hpp>
#include <ttl/tensor>

namespace stdml::datasets
{
template <typename Images, typename Labels>
struct basic_data_set {
    const Images images;
    const Labels labels;
};

template <typename Images, typename Labels>
auto make_dataset(Images &images, Labels &labels)
{
    return basic_data_set<Images, Labels>{
        .images = std::move(images),
        .labels = std::move(labels),
    };
}

template <typename...>
struct basic_data_sets;

template <typename Train, typename Test>
struct basic_data_sets<Train, Test> {
    const Train train;
    const Test test;
};

inline std::string safe_getenv(const std::string &name)
{
    const char *p = std::getenv(name.c_str());
    if (p) { return std::string(p); }
    return "";
}

inline std::string data_dir(const std::string &name = "")
{
    std::string prefix = safe_getenv("HOME") + "/var/data";
    if (!name.empty()) { prefix += "/" + name; }
    return prefix;
}

class mnist
{
    using data_set = basic_data_set<ttl::tensor<uint8_t, 3>,  //
                                    ttl::tensor<uint8_t, 1>>;

    using datasets = basic_data_sets<data_set, data_set>;

    template <typename R, ttl::rank_t r>
    static void read_tensor(const std::string &path,
                            const ttl::tensor_ref<R, r> &x)
    {
        namespace fs = std::filesystem;
        const std::string filename = fs::path(path).filename();
        {
            std::ifstream fin(path);
            if (!fin.is_open()) {
                throw std::runtime_error(path + " not found");
            }
        }
        (ttl::nn::ops::readfile(path))(x);
    }

  public:
    static constexpr auto shape = ttl::make_shape(28, 28);
    static constexpr uint8_t n_categories = 10;

    // http://yann.lecun.com/exdb/mnist/
    static data_set load(const std::string &name, const std::string &data_dir)
    {
        const int n = [name] {
            if (name == "train") {
                return 60 * 1000;
            } else if (name == "t10k") {
                return 10 * 1000;
            } else {
                throw std::invalid_argument("mnist name must be train or t10k");
            };
        }();
        ttl::tensor<uint8_t, 3> images(ttl::batch(n, shape));
        ttl::tensor<uint8_t, 1> labels(n);

        const std::string prefix = data_dir + "/" + name;
        read_tensor(prefix + "-images-idx3-ubyte", ttl::ref(images));
        read_tensor(prefix + "-labels-idx1-ubyte", ttl::ref(labels));
        return make_dataset(images, labels);
    }

    // load from default data_dir
    static data_set load(const std::string &name)
    {
        return load(name, data_dir("mnist"));
    }

    static datasets load_all(const std::string &data_dir)
    {
        return mnist::datasets{
            .train = load("train", data_dir),
            .test = load("t10k", data_dir),
        };
    }

    static datasets load_all() { return load_all(data_dir("mnist")); }
};
}  // namespace stdml::datasets
