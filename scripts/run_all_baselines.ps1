$ErrorActionPreference = "Stop"

$methods = @(
  "majority",
  "lexical_overlap",
  "similarity_only",
  "nli_only",
  "ess_rule",
  "sc_gate_only",
  "scad_rag"
)

foreach ($method in $methods) {
  python -m scad_rag.cli.run_pipeline --config configs/quick_test.yaml --method $method
}
