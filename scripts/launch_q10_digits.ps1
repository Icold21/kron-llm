$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\launch_digits.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== launch digits $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# 1. wait for the retokenisation
$deadline = (Get-Date).AddHours(3)
while ($true) {
    if ((Test-Path "data\bilingual\tokenize_digits.log") -and
        ((Get-Content "data\bilingual\tokenize_digits.log" -Raw) -match '\[done\] data')) { break }
    if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for tokenizer"; exit 1 }
    Start-Sleep -Seconds 60
}
Say "tokenizer done"

# 2. gate
python -u scripts\check_corpus_gate.py 2>&1 | Out-File -Encoding utf8 -Append $log
$gate = $LASTEXITCODE
Say ("gate exit " + $gate)
if ($gate -ne 0) { Say "GATE FAILED"; exit 2 }

# 3. stage and upload the new bins
$stage = Join-Path $root "data\kaggle_corpus"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item "data\bilingual\train.bpe.bin", "data\bilingual\val.bpe.bin" "$stage\" -Force
Copy-Item "data\bilingual\tokenizer.json" "$stage\" -Force
Copy-Item "data\bilingual\tokenizer_meta.json" "$stage\" -Force
python scripts\make_kaggle_meta.py "$stage\dataset-metadata.json" 2>&1 | Out-File -Encoding utf8 -Append $log
Say "uploading dataset version"
python -m kaggle datasets version -p $stage -m "digit-isolated tokenizer: every digit is its own token" 2>&1 | Out-File -Encoding utf8 -Append $log
$kag = $LASTEXITCODE
if ($kag -ne 0) {
    Say "version failed, trying create (new slug would be needed)"
    python -m kaggle datasets create -p $stage 2>&1 | Out-File -Encoding utf8 -Append $log
    $kag = $LASTEXITCODE
}
Say ("kaggle exit " + $kag)

# 4. kernel starts from scratch: code v9 + corpus only
python -c "import json; p=r'kaggle/q10/kernel-metadata.json'; d=json.load(open(p)); d['dataset_sources']=['icold21/kron-llm-code-v9','icold21/kron-corpus-2b']; json.dump(d, open(p,'w'), indent=2)"
python -m kaggle kernels push -p kaggle/q10 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("kernel push exit " + $LASTEXITCODE)
Say "DIGITS RUN PUSHED"
