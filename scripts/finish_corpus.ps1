$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\bilingual\finish_corpus.log"
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"
$session = "ses_e9b13e036ffe6nMU1JPmzLYbmi"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== finish corpus v3 $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# ------------------------------------------------ 1. wait for identity v7
$deadline = (Get-Date).AddHours(7)
while ($true) {
    if ((Test-Path "data\bilingual\identity_v7.log") -and
        ((Get-Content "data\bilingual\identity_v7.log" -Raw) -match [regex]::Escape("IDENTITY V7 DONE"))) { break }
    if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for identity v7"; break }
    Start-Sleep -Seconds 60
}
Say "identity v7 finished"

# ------------------------------------------------------ 2. copy identity out
Copy-Item "data\bilingual\shards\identity_*.jsonl" "data\identity\" -Force
Copy-Item "data\identity\*.jsonl" "C:\Users\Icold\Desktop\kron_corpus\data\identity\" -Force
Copy-Item "data\bilingual\manifest.jsonl" "C:\Users\Icold\Desktop\kron_corpus\manifest.jsonl" -Force
Say "identity shards and manifest copied"

# --------------------------------------------- 3. append the Russian bulk
if (Test-Path "data\bilingual\shards\wikipedia_ru_bulk.jsonl") {
    python -u scripts\add_bulk_to_mix.py --shard data\bilingual\shards\wikipedia_ru_bulk.jsonl --lang ru --domain encyclopedic 2>&1 | Out-File -Encoding utf8 -Append $log
    Say ("bulk wikipedia exit " + $LASTEXITCODE)
}
if (Test-Path "data\bilingual\shards\wikisource_ru_bulk.jsonl") {
    python -u scripts\add_bulk_to_mix.py --shard data\bilingual\shards\wikisource_ru_bulk.jsonl --lang ru --domain literature 2>&1 | Out-File -Encoding utf8 -Append $log
    Say ("bulk wikisource exit " + $LASTEXITCODE)
}

# --------------------------------------------------------- 4. patch identity
python -u scripts\patch_identity_mix.py --lang en --lang ru 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("patch exit " + $LASTEXITCODE)

# ------------------------------------------------------ 5. tokenizer + bins
Say "tokenizer 8K + bins (final)"
python -u scripts\prepare_bilingual_bin.py --vocab 8192 2>&1 | Out-File -Encoding utf8 -Append $log
$tokExit = $LASTEXITCODE
Say ("tokenizer exit " + $tokExit)

# ------------------------------------------------------------- 6. hard gate
$gateExit = 1
if ($tokExit -eq 0) {
    python -u scripts\check_corpus_gate.py 2>&1 | Out-File -Encoding utf8 -Append $log
    $gateExit = $LASTEXITCODE
    Say ("gate exit " + $gateExit)
}

# --------------------------------------------------------- 7. Kaggle upload
$kaggleExit = -1
if ($gateExit -eq 0) {
    $stage = Join-Path $root "data\kaggle_corpus"
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item "data\bilingual\train.bpe.bin", "data\bilingual\val.bpe.bin" "$stage\" -Force
    Copy-Item "data\bilingual\tokenizer.json" "$stage\" -Force
    Copy-Item "data\bilingual\tokenizer_meta.json" "$stage\" -Force
    python scripts\make_kaggle_meta.py "$stage\dataset-metadata.json" 2>&1 | Out-File -Encoding utf8 -Append $log
    Say "uploading a new dataset version"
    python -m kaggle datasets version -p $stage -m "identity v7 + full ruwiki/wikisource bulk: >=2B tokens, RU >=30%" 2>&1 | Out-File -Encoding utf8 -Append $log
    $kaggleExit = $LASTEXITCODE
    if ($kaggleExit -ne 0) {
        Say "version failed, trying create"
        python -m kaggle datasets create -p $stage 2>&1 | Out-File -Encoding utf8 -Append $log
        $kaggleExit = $LASTEXITCODE
    }
    Say ("kaggle exit " + $kaggleExit)
} else {
    Say "gate failed: bins are NOT uploaded"
}

# --------------------------------------------------------------- 8. report
python -u scripts\final_report.py 2>&1 | Out-File -Encoding utf8 -Append $log
$rep = Join-Path $root "data\bilingual\final_report.md"
Add-Content -Encoding utf8 $rep @"

## Статус

- identity v7: завершён
- токенизация: exit $tokExit
- гейт (>=2.0B токенов, RU >=30 %): exit $gateExit
- Kaggle: exit $kaggleExit
- лог: ``data\bilingual\finish_corpus.log``
"@

python $mail send --model "DeepSeek V4.1 Flash Free" --session $session --project "kron_llm" --subject "KronCorpus: итог сборки (2B гейт)" --body-file $rep 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("email exit " + $LASTEXITCODE)
Say "FINAL DONE"
