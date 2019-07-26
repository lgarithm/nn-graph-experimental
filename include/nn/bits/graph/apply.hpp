#pragma once
#define NN_GRAPG_TRACE 1

#include <experimental/iterator>
#include <experimental/reflect>
#include <functional>
#include <sstream>

#if NN_GRAPG_TRACE
#include <nn/bits/graph/trace.hpp>
#endif

template <bool> struct maybe_apply;

template <> struct maybe_apply<false> {

    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
        static constexpr auto arity = sizeof...(Args);
        std::array<std::string, arity> names(
            {demangled_type_info_name(typeid(Args))...});

        std::stringstream ss;
        ss << "(\n\t";
        std::copy(names.begin(), names.end(),
                  std::experimental::make_ostream_joiner(ss, "\n\t"));
        ss << "\n)";

        std::cerr << "can't apply " << demangled_type_info_name(typeid(F))
                  << " to " << ss.str() << std::endl;
    }
};

template <> struct maybe_apply<true> {
    template <typename Tuple, size_t... I>
    static std::string signature(const Tuple &args, std::index_sequence<I...>)
    {
        std::array<std::string, sizeof...(I)> names(
            {ttl::tensor_type_name(std::get<I>(args))...});
        std::string name;
        for (auto n : names) {
            if (!name.empty()) { name += ","; }
            name += n;
        }
        return name;
    }

    template <typename F, typename... Args>
    void operator()(const F &f, const std::tuple<Args...> &args) const
    {
        const auto f_name = demangled_type_info_name(typeid(F));
        // static constexpr auto arity = sizeof...(Args);
        // const auto name = f_name + "(" +
        //                   signature(args, std::make_index_sequence<arity>())
        //                   +
        //                   ")";
#if NN_GRAPG_TRACE
        TRACE_SCOPE(f_name);
#endif
        std::apply(f, args);
    }
};

template <typename F, typename... Args>
void apply_if(const F &f, const std::tuple<Args...> &args)
{
    maybe_apply<std::is_invocable<F, Args...>::value>()(f, args);
}
