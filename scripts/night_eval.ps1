# After the single session: evaluate, report, mail, and power the machine down.
#
# This is the part the owner asked for explicitly: the curve, the metrics and
# the generation results for the raw checkpoint AND for the averaged one, one
# email, then shut down.  It runs whether the session succeeded or failed -- a
# silent failure overnight is worse than a bad report.
#
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI, so any non-ASCII here becomes
# mojibake in the log and in the subject line.
param(
    [switch]$Shutdown
)
$ErrorActionPreference = "Continue"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "data\night_eval.log"
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"

function Say([string]$msg) {
    $line = ("{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $msg)
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

$last = Join-Path $root "data\kaggle_out\q10\last.pt"
$soup = Join-Path $root "data\kaggle_out\q10\soup.pt"
$md = Join-Path $root "data\report_session.md"
$html = Join-Path $root "data\report_session.html"

Say "night eval start"

if (-not (Test-Path $last)) {
    Say "no checkpoint downloaded: the session produced nothing"
    "# Q10: the session produced no checkpoint`n`nThe downloaded output does not contain ``last.pt``." |
        Out-File -Encoding utf8 $md
} else {
    Say "evaluating last.pt"
    python scripts\eval_samples.py --ckpt $last --mode all --out data\eval_last.json 2>&1 |
        Select-Object -Last 3 | ForEach-Object { Say ("  " + $_) }

    if (Test-Path $soup) {
        Say "evaluating soup.pt"
        python scripts\eval_samples.py --ckpt $soup --mode all --out data\eval_soup.json 2>&1 |
            Select-Object -Last 3 | ForEach-Object { Say ("  " + $_) }
    } else {
        Say "no soup.pt (the soup step may have been skipped)"
    }

    Say "assembling the report"
    python scripts\report_session.py --out data\report_session.md 2>&1 |
        Select-Object -Last 2 | ForEach-Object { Say ("  " + $_) }
}

python scripts\md_to_html.py $md $html 2>&1 | Select-Object -Last 2 |
    ForEach-Object { Say ("  " + $_) }

if (Test-Path $html) {
    $step = "unknown"
    if (Test-Path $last) {
        $step = (python -c "import torch; ck=torch.load(r'$last',map_location='cpu',weights_only=False); print(ck.get('step'))" 2>$null | Select-Object -Last 1)
    }
    $subject = "Q10 session: step $step, curve and samples (last + soup)"
    Say "mailing: $subject"
    1..5 | ForEach-Object {
        python $mail send --model "DeepSeek V4.1 Flash" --project "kron_llm" `
            --session "kron_chain" --subject $subject --body-file $html --html 2>&1 |
            Select-Object -Last 1 | ForEach-Object { Say ("  " + $_) }
        Start-Sleep -Seconds 60
    }
} else {
    Say "no HTML report to mail"
}

Say "night eval done"

if ($Shutdown) {
    Say "shutting down in 2 minutes"
    shutdown /s /t 120 /c "KronLLM night run finished"
}
