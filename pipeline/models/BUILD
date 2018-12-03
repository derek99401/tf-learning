py_library(
    name = "model_interface",
    visibility = ["//visibility:public"],
    srcs = ["model_interface.py"],
)

py_library(
    name = "base_model",
    visibility = ["//visibility:public"],
    srcs = ["base_model.py"],
    deps = [
        ":model_interface",
    ],
)

py_library(
    name = "input_only_model",
    visibility = ["//visibility:public"],
    srcs = ["input_only_model.py"],
    deps = [
        ":base_model",
    ],
)
