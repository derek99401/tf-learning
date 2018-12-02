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
        "//tfpipeline/models:input_only_model",
        ":tfpipeline",
    ],
)
