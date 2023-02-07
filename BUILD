load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
)

tf_cc_shared_object(
    name = "_XLAC.so",
    copts = [
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_XLAC",
        "-Iexternal/torch/torch/include",
    ],
    linkopts = ["-Wl,-Bsymbolic"],
    visibility = ["//visibility:public"],
    deps = [
        "//torch_xla/csrc:init_python_bindings",
        "@libtorch",
     	"@torch//:headers",
        "@libtorch//:libtorch_cpu",
        "@libtorch//:libtorch_python",
        "@pybind11",
        "@python3_9//:python_headers",
    ],
)
