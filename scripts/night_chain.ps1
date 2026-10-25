$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\chain_night.log"
$idlog = Join-Path $root "data\bilingual\identity_v5.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== night chain start ===" | Out-File -Encoding utf8 $log
Say "waiting for identity v5 (log must contain: rick done)"

# ---------------------------------------------------------------- 1. identity
$deadline = (Get-Date).AddHours(11)
$finished = $false
while (-not $finished) {
    if (Test-Path $idlog) {
        $txt = Get-Content $idlog -Raw -ErrorAction SilentlyContinue
        if ($txt -match [regex]::Escape("=== rick done")) { $finished = $true }
    }
    if (-not $finished) {
        if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for identity"; exit 1 }
        Start-Sleep -Seconds 60
    }
}
Say "identity run reported done"

# sanity: all four shards present with the expected number of documents
foreach ($spec in @(@("identity_en", 800), @("identity_ru", 800),
                    @("identity_rick_en", 850), @("identity_rick_ru", 850))) {
    $p = Join-Path $root ("data\bilingual\shards\{0}.jsonl" -f $spec[0])
    if (-not (Test-Path $p)) { Say ("MISSING " + $spec[0]); exit 1 }
    $n = (Get-Content $p | Measure-Object -Line).Lines
    Say ("{0}: {1} docs" -f $spec[0], $n)
    if ($n -lt $spec[1]) { Say ("SHORT " + $spec[0]); exit 1 }
}

# --------------------------------------------------------- 1b. identity report
Say "writing identity report and emailing it"
python -u scripts\identity_report.py 2>&1 | Out-File -Encoding utf8 -Append $log
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"
$subject = "Identity готов: 800 обычных + 850 пастиш на язык, без цифр"
python $mail send --model "DeepSeek V4.1 Flash Free" --session "ses_e9b13e036ffe6nMU1JPmzLYbmi" --project "kron_llm" --subject $subject --body-file "$root\data\bilingual\identity_report.md" 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("identity email exit " + $LASTEXITCODE)

# ---------------------------------------------------------------- 2. copy
Copy-Item "$root\data\bilingual\shards\identity_*.jsonl" "$root\data\identity\" -Force
Copy-Item "$root\data\identity\*.jsonl" "C:\Users\Icold\Desktop\kron_corpus\data\identity\" -Force
Copy-Item "$root\data\bilingual\manifest.jsonl" "C:\Users\Icold\Desktop\kron_corpus\manifest.jsonl" -Force
Say "identity shards copied"

# ---------------------------------------------------------------- 3. build
Say "build: dedup + quotas + held-out"
python -u scripts\build_bilingual.py build --budget-en-mb 4300 --budget-ru-mb 2870 --heldout-mb 120 2>&1 | Out-File -Encoding utf8 -Append $log
if ($LASTEXITCODE -ne 0) { Say "BUILD FAILED"; exit 2 }
Say "build done"

# ---------------------------------------------------------- 4. tokenizer + bins
Say "tokenizer 8K + bins"
python -u scripts\prepare_bilingual_bin.py --vocab 8192 2>&1 | Out-File -Encoding utf8 -Append $log
if ($LASTEXITCODE -ne 0) { Say "TOKENIZER FAILED"; exit 3 }

$train = Join-Path $root "data\bilingual\train.bpe.bin"
$val = Join-Path $root "data\bilingual\val.bpe.bin"
if (-not (Test-Path $train)) { Say "no train.bpe.bin"; exit 4 }
$mb = [math]::Round((Get-Item $train).Length / 1MB, 1)
Say ("bins ready: train {0} MB" -f $mb)
if ($mb -lt 500) { Say "train bin too small, not uploading"; exit 5 }

# --------------------------------------------------------- 5. Kaggle dataset
$stage = Join-Path $root "data\kaggle_corpus"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item $train, $val "$stage\" -Force
Copy-Item (Join-Path $root "data\bilingual\tokenizer.json") "$stage\" -Force
Copy-Item (Join-Path $root "data\bilingual\tokenizer_meta.json") "$stage\" -Force
python scripts\make_kaggle_meta.py "$stage\dataset-metadata.json" 2>&1 | Out-File -Encoding utf8 -Append $log
Say "uploading bins to Kaggle (icold21/kron-corpus-2b)"
python -m kaggle datasets create -p $stage 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("kaggle exit " + $LASTEXITCODE)

Say "CHAIN DONE"
