param(
  [string[]]$Instances = @("pr2392","fnl4461","pla33810"),
  [string]$Tag = "auto",
  [int]$Workers = 4
)
$env:OMP_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"

python scripts\run_large_cli.py `
  --instances $Instances `
  --tsplib_dir data\tsplib `
  --config configs\params_z3000.yaml `
  --seeds 42 123 456 789 999 `
  --workers $Workers `
  --results_dir results `
  --tag $Tag