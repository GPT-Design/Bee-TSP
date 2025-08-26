# list all
Get-Content update_history\manifest.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Format-Table id,date,commit,branch

# latest entry
(Get-Content update_history\manifest.jsonl | Select-Object -Last 1) | ConvertFrom-Json

# filter by id
(Get-Content update_history\manifest.jsonl | Where-Object { $_ -match '"id":\s*5' }) | ConvertFrom-Json
