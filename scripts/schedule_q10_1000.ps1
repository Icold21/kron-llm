$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_schedule.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== 10:00 report scheduled $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

$target = (Get-Date).Date.AddHours(10)
if ((Get-Date) -gt $target) { $target = $target.AddDays(1) }
Say ("waiting until " + $target.ToString("HH:mm"))

while ((Get-Date) -lt $target) {
    $left = ($target - (Get-Date)).TotalSeconds
    if ($left -gt 120) { Start-Sleep -Seconds 60 } else { Start-Sleep -Seconds 5 }
}

# If the chain already mailed a session report in the last 90 minutes, another
# identical letter is noise.
$marker = Join-Path $root "data\last_mail.txt"
if (Test-Path $marker) {
    $last = [datetime]::Parse((Get-Content $marker -Raw -Encoding utf8).Trim())
    if (((Get-Date) - $last).TotalMinutes -lt 90) {
        Say ("session report already sent at " + $last.ToString("HH:mm") + ", skipping")
        exit 0
    }
}

# The session that ended overnight has its log.csv downloaded by now; if the
# chain is still working, wait for it rather than mailing an empty report.
$deadline = (Get-Date).AddMinutes(90)
while (-not (Test-Path "data\kaggle_out\q10\log.csv")) {
    if ((Get-Date) -gt $deadline) { Say "no log.csv by the deadline, mailing what we have"; break }
    Start-Sleep -Seconds 60
}

Say "sending the 10:00 report"
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\email_q10_report.ps1")
Say "done"
