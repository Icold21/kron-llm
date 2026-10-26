$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_resume.log"
$ckpt = Join-Path $root "data\kaggle_out\q10\last.pt"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== resume watcher $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# 1. wait for a complete checkpoint file (size stable for 30 s, > 50 MB)
$deadline = (Get-Date).AddHours(12)
$lastSize = -1
while ($true) {
    if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for last.pt"; exit 1 }
    if (Test-Path $ckpt) {
        $size = (Get-Item $ckpt).Length
        if ($size -gt 50MB -and $size -eq $lastSize) { break }
        $lastSize = $size
    }
    Start-Sleep -Seconds 30
}
Say ("checkpoint complete: " + [math]::Round((Get-Item $ckpt).Length / 1MB, 1) + " MB")

# 2. stage it as a dataset
$stage = Join-Path $root "data\kaggle_ckpt"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item $ckpt (Join-Path $stage "last.pt") -Force
python -c "import json; json.dump({'title':'Kron Q10 checkpoint','id':'icold21/kron-q10-ckpt','licenses':[{'name':'other'}]}, open(r'$stage\dataset-metadata.json','w'), indent=2)"
python -m kaggle datasets version -p $stage -m "session checkpoint" 2>&1 | Out-File -Encoding utf8 -Append $log
if ($LASTEXITCODE -ne 0) {
    Say "version failed, creating the checkpoint dataset"
    python -m kaggle datasets create -p $stage 2>&1 | Out-File -Encoding utf8 -Append $log
}
Say ("checkpoint dataset exit " + $LASTEXITCODE)

# 3. attach it to the kernel and push the next session
python -c "import json; p=r'kaggle/q10/kernel-metadata.json'; d=json.load(open(p)); d['dataset_sources']=['icold21/kron-llm-code-v8','icold21/kron-corpus-2b','icold21/kron-q10-ckpt']; json.dump(d, open(p,'w'), indent=2)"
python -m kaggle kernels push -p kaggle/q10 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("kernel push exit " + $LASTEXITCODE)
Say "SESSION 2 PUSHED"
