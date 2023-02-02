"""PyTorch/XLA rules including PyTorch headers and libraries."""

def ptxla_cc_library(
        copts = [],
        deps = [],
        **kwargs):
    extra_copts = ["-Iexternal/torch/torch/include"]
    extra_deps = ["@torch//:headers"]
    native.cc_library(
        copts = copts + extra_copts,
        deps = deps + extra_deps,
        **kwargs
    )
