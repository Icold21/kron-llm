# Start the 03:05 session guard, run exactly one session, evaluate it, mail and
# shut down.
#
# Two reasons this is a separate entry point rather than a second trigger on
# kron_chain:
#
# * The corpus rebuild finished an hour earlier and is still uploading.  A
#   kernel that starts while its dataset version is mid-upload fails at mount,
#   which costs a whole session -- so this waits for the "uploaded and ready"
#   flag first, and says so in the log if it never appears.
# * The owner asked for one session and then the results, not for an unattended
#   chain that keeps spending quota: the chain runs with -Once -NoMail and the
#   report is assembled here, from the same files.
#
# ASCII only: PowerShell 5.1 reads .ps1 as ANSI.
$ErrorActionPreference = "Continue"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "data\session_guard.log"
$flag = Join-Path $root "data\corpus_ready.flag"

function Say([string]$msg) {
    $line = ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
    Add-Content -Path $log -Value $line -Encoding utf8
}

Say "guard fired; waiting for the corpus flag"

$waited = 0
while (-not (Test-Path $flag) -and $waited -lt 3600) {
    Start-Sleep -Seconds 300
    $waited += 300
    Say ("still waiting for corpus_ready.flag ({0} min)" -f [int]($waited / 60))
}

if (Test-Path $flag) {
    $info = (Get-Content $flag -Raw) -replace "`r?`n", " "
    Say ("corpus ready: " + $info)
} else {
    # The owner's rule: if the corpus was not polished in time, do not start a
    # session on a half-ready dataset.  Report and stop instead.
    Say "corpus flag never appeared: NOT starting a session"
    $md = Join-Path $root "data\report_session.md"
    ("# Q10: session not started`n`nThe rebuilt corpus was not uploaded and " +
     "marked ready before 03:05, so no session was launched. Nothing was spent.") |
        Out-File -Encoding utf8 $md
    $html = Join-Path $root "data\report_session.html"
    python scripts\md_to_html.py $md $html 2>&1 | ForEach-Object { Say ("  " + $_) }
    $mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"
    python $mail send --model "DeepSeek V4.1 Flash" --project "kron_llm" `
        --session "kron_chain" --subject "Q10: session not started (corpus not ready)" `
        --body-file $html --html 2>&1 | Select-Object -Last 1 |
        ForEach-Object { Say ("  " + $_) }
    Say "shutting down in 2 minutes"
    shutdown /s /t 120 /c "KronLLM: corpus not ready, no session"
    exit 1
}

Say "starting chain_q10.ps1 -Once -NoMail"
& (Join-Path $root "scripts\chain_q10.ps1") -Once -NoMail
Say "chain returned"

Say "running night_eval.ps1 -Shutdown"
& (Join-Path $root "scripts\night_eval.ps1") -Shutdown
