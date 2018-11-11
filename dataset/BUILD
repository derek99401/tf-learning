py_binary(
    name = "build_random_dataset",
    srcs = ["build_random_dataset.py"],
    deps = [
        ":lmdb_dataset_builder",
        ":dataset_utils"
    ],
)

py_library(
    name = "lmdb_dataset_builder",
    srcs = ["lmdb_dataset_builder.py"],
)

py_library(
    name = "dataset_utils",
    srcs = ["utils.py"],
)
