$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_chain.log"
$kernel = "icold21/kron-q10-10m-bilingual-pretraining"
$code = "icold21/kron-llm-code-v11"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

function Mail($subject, $body) {
    python "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py" send `
        --model "DeepSeek V4.1 Flash Free" --project "kron_llm" `
        --subject $subject --body-file $body 2>&1 | Out-File -Encoding utf8 -Append $log
}

"=== chain started $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# 1. wait for the session to end.  A transient API failure must not look like
# completion, otherwise the next session would be pushed on top of a live run.
$deadline = (Get-Date).AddHours(14)
while ($true) {
    if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for the session"; exit 1 }
    $status = python -m kaggle kernels status $kernel 2>&1 | Select-Object -Last 1
    Say $status
    if ($status -match "COMPLETE|ERROR|CANCEL|FAILED") { break }
    Start-Sleep -Seconds 300
}

# 2. download the session output (checkpoints, log.csv, results.json)
$out = Join-Path $root "data\kaggle_out"
New-Item -ItemType Directory -Force -Path $out | Out-Null
python -m kaggle kernels output $kernel -p $out 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("output exit " + $LASTEXITCODE)

# 3. report the curve and mail it
Remove-Item "data\q10_epoch_done.flag" -ErrorAction SilentlyContinue
python scripts\report_q10.py --tag session 2>&1 | Out-File -Encoding utf8 -Append $log
$body = Join-Path $root "data\report_q10.md"
$subjectFile = Join-Path $root "data\report_q10_subject.txt"
$subject = if (Test-Path $subjectFile) { (Get-Content $subjectFile -Raw).Trim() } else { "Q10 session report" }
if (Test-Path $body) { Mail $subject $body }

# 4. publish the checkpoint so the next session can attach it
$ckpt = Join-Path $root "data\kaggle_out\q10\last.pt"
if (-not (Test-Path $ckpt)) {
    Say "no last.pt in the output; stopping before the resume"
    exit 2
}
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

# 5. the epoch is complete: keep the weights, do not push another training
#    session -- the WSD decay pass is launched separately, on curated data.
if (Test-Path "data\q10_epoch_done.flag") {
    Say "EPOCH COMPLETE: checkpoint published, waiting for the decay pass"
    Say "NEXT: launch Q10A (decay, 20% of an epoch, LR 3e-4 -> 1e-6)"
    exit 0
}

# 6. otherwise push the next session: same code, same corpus, plus the checkpoint
python -c "import json; p=r'kaggle/q10/kernel-metadata.json'; d=json.load(open(p)); d['dataset_sources']=['$code','icold21/kron-corpus-2b','icold21/kron-q10-ckpt']; json.dump(d, open(p,'w'), indent=2)"
python -m kaggle kernels push -p kaggle/q10 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("kernel push exit " + $LASTEXITCODE)
Say "NEXT SESSION PUSHED"
