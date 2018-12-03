git_repository(
  name = "org_pubref_rules_protobuf",
  remote = "https://github.com/pubref/rules_protobuf",
  tag = "v0.8.2",
  #commit = "..." # alternatively, use latest commit on master
)

load("@org_pubref_rules_protobuf//python:rules.bzl", "py_proto_repositories")
py_proto_repositories()

load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cpp_proto_repositories")
cpp_proto_repositories()
