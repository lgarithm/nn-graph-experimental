#pragma once
#include <cxxabi.h>

static inline std::string demangled_type_info_name(const std::type_info &ti)
{
    int status = 0;
    return abi::__cxa_demangle(ti.name(), 0, 0, &status);
}
