
py_library(
    name = "tfpipeline",
    srcs = ["tfpipeline.py"],
    deps = [
    ],
)

py_binary(
    name = "run_pipeline",
    srcs = ["run_pipeline.py"],
    deps = [
        "//pipeline/models:input_only_model",
        ":tfpipeline",
    ],
)

py_binary(
    name = "print_config",
    srcs = ["print_config.py"],
    deps = [
        "//pipeline/proto:tfpipeline_options_py",
    ],
)
