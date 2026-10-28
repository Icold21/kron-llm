# Send the session reports the chain queued overnight.
#
# A report that lands at 03:00 is read half-asleep or not at all, so the chain no
# longer mails it immediately: it appends the report to data/report_mail_pending.json
# and this script, fired by the one-off task `kron_report_mail`, sends everything
# queued at 08:00.  Failures are never queued -- an error mail goes out at once.
#
# The queue is a JSON array so a second session ending before 08:00 cannot
# silently replace the first report.  Entries are removed as they are sent, so a
# failed send keeps the rest for the next attempt instead of losing them.

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root

$log = Join-Path $root "data\q10_chain.log"
$pending = Join-Path $root "data\report_mail_pending.json"
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"

function Say($msg) {
    ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) | Out-File -Encoding utf8 -Append $log
}

if (-not (Test-Path $pending)) {
    Say "queued report: nothing pending"
    exit 0
}

$queue = @()
if (Test-Path $pending) {
    # ConvertFrom-Json hands back a JSON array as one object: @(...) around it
    # would nest the queue inside a single entry and "send" a broken item.
    $parsed = Get-Content $pending -Raw -Encoding utf8 | ConvertFrom-Json
    if ($parsed -is [System.Array]) { $queue = $parsed } elseif ($null -ne $parsed) { $queue = @($parsed) }
}
Say ("queued report: " + $queue.Count + " to send")

$left = @()
foreach ($item in $queue) {
    $html = Join-Path $root $item.html
    if (-not (Test-Path $html)) {
        Say ("queued report: " + $item.html + " is missing, dropped")
        continue
    }
    $out = python $mail send --model "DeepSeek V4.1 Flash" --project "kron_llm" `
        --session "kron_chain" --subject $item.subject --body-file $html --html 2>&1
    $out | Out-File -Encoding utf8 -Append $log
    if ($LASTEXITCODE -eq 0) {
        Say ("queued report sent: " + $item.subject)
    } else {
        Say ("queued report FAILED: " + $item.subject + " (kept for a retry)")
        $left += $item
    }
}

if ($left.Count -eq 0) {
    Remove-Item $pending -Force -ErrorAction SilentlyContinue
    schtasks /Delete /TN "kron_report_mail" /F 2>&1 | Out-Null
} elseif ($left.Count -eq 1) {
    $left[0] | ConvertTo-Json | Out-File -Encoding utf8 $pending
} else {
    ConvertTo-Json -InputObject $left | Out-File -Encoding utf8 $pending
}
exit 0
