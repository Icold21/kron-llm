$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_watch.log"
$kernel = "icold21/kron-q10-10m-bilingual-pretraining"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== watching $kernel $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

while ($true) {
    $status = python -m kaggle kernels status $kernel 2>&1 | Select-Object -Last 1
    Say $status
    if ($status -notmatch "RUNNING") { break }
    Start-Sleep -Seconds 600
}

Say "run finished, downloading outputs"
$out = Join-Path $root "data\kaggle_out"
New-Item -ItemType Directory -Force -Path $out | Out-Null
python -m kaggle kernels output $kernel -p $out 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("output exit " + $LASTEXITCODE)
Say "Q10 RUN ENDED"
