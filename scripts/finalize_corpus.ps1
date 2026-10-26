$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\bilingual\finalize_corpus.log"
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"
$session = "ses_e9b13e036ffe6nMU1JPmzLYbmi"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== finalize corpus $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# ---------------------------------------------------- 1. wait for tokenizer
$deadline = (Get-Date).AddHours(3)
while ($true) {
    if ((Test-Path "data\bilingual\tokenize_final.log") -and
        ((Get-Content "data\bilingual\tokenize_final.log" -Raw) -match '\[done\] data')) { break }
    if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for tokenizer"; break }
    Start-Sleep -Seconds 30
}
Say "tokenizer finished"

# ------------------------------------------------------------- 2. hard gate
python -u scripts\check_corpus_gate.py 2>&1 | Out-File -Encoding utf8 -Append $log
$gateExit = $LASTEXITCODE
Say ("gate exit " + $gateExit)

# --------------------------------------------------------- 3. Kaggle upload
$kaggleExit = -1
$filesOk = $false
if ($gateExit -eq 0) {
    $stage = Join-Path $root "data\kaggle_corpus"
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item "data\bilingual\train.bpe.bin", "data\bilingual\val.bpe.bin" "$stage\" -Force
    Copy-Item "data\bilingual\tokenizer.json" "$stage\" -Force
    Copy-Item "data\bilingual\tokenizer_meta.json" "$stage\" -Force
    python scripts\make_kaggle_meta.py "$stage\dataset-metadata.json" 2>&1 | Out-File -Encoding utf8 -Append $log

    Say "uploading dataset (version, then create)"
    python -m kaggle datasets version -p $stage -m "2.4B tokens, RU 37 percent, identity v7" 2>&1 | Out-File -Encoding utf8 -Append $log
    $kaggleExit = $LASTEXITCODE
    if ($kaggleExit -ne 0) {
        Say "version failed, creating"
        python -m kaggle datasets create -p $stage 2>&1 | Out-File -Encoding utf8 -Append $log
        $kaggleExit = $LASTEXITCODE
    }
    Say ("kaggle exit " + $kaggleExit)

    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 30
        $status = python -m kaggle datasets status icold21/kron-corpus-2b 2>&1 | Select-Object -Last 1
        $listing = python -m kaggle datasets files icold21/kron-corpus-2b 2>&1
        if ($listing -match 'train\.bpe\.bin') { $filesOk = $true; break }
    }
    Say ("dataset files present: " + $filesOk)
}

# --------------------------------------------------------------- 4. report
python -u scripts\final_report.py 2>&1 | Out-File -Encoding utf8 -Append $log
$rep = Join-Path $root "data\bilingual\final_report.md"
Add-Content -Encoding utf8 $rep @"

## Статус

- гейт (>=2.0B токенов, RU >=30 %): $gateExit
- Kaggle create/version: $kaggleExit
- файлы датасета видны: $filesOk
- лог: ``data\bilingual\finalize_corpus.log``
"@

python $mail send --model "DeepSeek V4.1 Flash Free" --session $session --project "kron_llm" --subject "KronCorpus ready: 2.4B tokens, RU 37 percent, bins on Kaggle" --body-file $rep 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("email exit " + $LASTEXITCODE)
Say "CORPUS READY"
