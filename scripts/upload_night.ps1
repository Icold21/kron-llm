# Stage and upload the rebuilt corpus, then mark it ready.
#
# The 03:05 guard waits for data/corpus_ready.flag, and that flag is written
# only when both dataset versions exist on Kaggle and report status "ready".  A
# kernel that starts while a dataset version is still processing fails at mount
# and burns a whole session, so the flag has to mean "verified", not "uploaded".
#
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "data\upload_night.log"
$flag = Join-Path $root "data\corpus_ready.flag"

function Say([string]$msg) {
    $line = ("{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $msg)
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Kaggle([string[]]$argv, [int]$tries = 4) {
    for ($i = 1; $i -le $tries; $i++) {
        $out = (& python -m kaggle @argv 2>&1) -join "`n"
        if ($out -notmatch "SSLError|ConnectionError|Max retries|Read timed out") {
            return $out
        }
        Say ("  kaggle retry {0}/{1} ({2})" -f $i, $tries, $argv[0])
        Start-Sleep -Seconds (10 * $i)
    }
    return $out
}

Remove-Item $flag -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 1. wait for the build
$bin = "E:\kron_bins\epoch\train.bpe.bin"
Say "waiting for $bin"
$lastSize = -1
$stable = 0
while ($true) {
    if (Test-Path $bin) {
        $size = (Get-Item $bin).Length
        if ($size -eq $lastSize -and $size -gt 100MB) {
            $stable += 1
            if ($stable -ge 3) { break }        # 3 x 60 s unchanged
        } else {
            $stable = 0
            $lastSize = $size
        }
    }
    Start-Sleep -Seconds 60
}
Say ("build finished: {0:N0} bytes" -f (Get-Item $bin).Length)

# The metadata is what says *which* corpus this is.  A stale bin has a plausible
# size and the same name, so the build is not trusted until the metadata records
# the full-prompt rebuild.
$meta = "E:\kron_bins\epoch\tokenizer_meta.json"
# phase rate prompted already docs tokens -- one line, easy to check in a log.
$check = python -c "import json; m=json.load(open(r'$meta')); d=m.get('domain_prompts') or {}; print(m.get('phase'), d.get('rate'), d.get('prompted_docs'), d.get('already_prompted_docs'), d.get('docs'), m.get('train_tokens'))" 2>&1 | Select-Object -Last 1
Say ("epoch metadata: " + $check)
$parts = ($check -split '\s+')
if ($parts.Count -lt 6) { Say "cannot read the metadata; not uploading"; exit 1 }
$covered = [int64]$parts[2] + [int64]$parts[3]
if ($parts[0] -ne "epoch" -or [double]$parts[1] -ne 1.0 -or $covered -ne [int64]$parts[4]) {
    Say ("corpus is not fully prompted: {0} prompted + {1} already = {2} of {3}" -f $parts[2], $parts[3], $covered, $parts[4])
    exit 1
}
Say ("corpus verified: all {0} documents carry a system prompt ({1} added, {2} already chat)" -f $parts[4], $parts[2], $parts[3])

# ---------------------------------------------------------------- 2. stage + upload the epoch corpus
$stage = "E:\kron_bins\kaggle_corpus"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($f in @("train.bpe.bin", "val.bpe.bin", "tokenizer.json",
                 "tokenizer_meta.json")) {
    $src = Join-Path "E:\kron_bins\epoch" $f
    if (-not (Test-Path $src)) { Say ("[miss] " + $f); continue }
    Copy-Item $src (Join-Path $stage $f) -Force
}
if (Test-Path "data\bilingual\val_index.json") {
    Copy-Item "data\bilingual\val_index.json" (Join-Path $stage "val_index.json") -Force
}
$total = [math]::Round(((Get-ChildItem $stage | Measure-Object Length -Sum).Sum) / 1GB, 2)
Say ("staged epoch corpus: {0} GB" -f $total)
python scripts\make_kaggle_meta.py (Join-Path $stage "dataset-metadata.json") 2>&1 |
    ForEach-Object { Say ("  " + $_) }

Say "uploading epoch corpus"
$out = Kaggle @("datasets", "version", "-p", $stage, "-m",
                "epoch bin rebuilt: chemistry, textbooks, DIY, system prompts", "--dir-mode", "zip")
Say ("  " + (($out -split "`n") | Select-Object -Last 1))
if ($out -match "not found|does not exist|404") {
    Say "version failed; creating the dataset"
    $out = Kaggle @("datasets", "create", "-p", $stage)
    Say ("  " + (($out -split "`n") | Select-Object -Last 1))
}

# ---------------------------------------------------------------- 3. upload the anneal bin
Say "staging the anneal bin"
python scripts\make_anneal_dataset.py 2>&1 | Select-Object -Last 3 |
    ForEach-Object { Say ("  " + $_) }
Say "uploading the anneal bin"
$out2 = Kaggle @("datasets", "version", "-p", "data\kaggle_anneal_bin", "-m",
                 "decay bin rebuilt: chemistry + textbooks + DIY, prompts on")
Say ("  " + (($out2 -split "`n") | Select-Object -Last 1))
if ($out2 -match "not found|does not exist|404") {
    Say "version failed; creating the anneal dataset"
    $out2 = Kaggle @("datasets", "create", "-p", "data\kaggle_anneal_bin")
    Say ("  " + (($out2 -split "`n") | Select-Object -Last 1))
}

# ---------------------------------------------------------------- 4. verify ready
# Each dataset only has to be seen ready once: on this link the API fails about
# half the calls, so requiring a single poll where both answer ready never
# happens even when both have finished.
Say "verifying both datasets"
python scripts\verify_datasets.py 2>&1 | ForEach-Object { Say ("  " + $_) }
if (Test-Path $flag) {
    Say "corpus_ready.flag written"
    exit 0
}
Say "datasets did not verify as ready; no flag, the 03:05 guard will refuse to start"
exit 1
