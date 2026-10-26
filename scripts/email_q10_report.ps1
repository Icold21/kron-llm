$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_email.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

# 1. rebuild the report from whatever log.csv the chain has downloaded
python scripts\report_q10.py --tag status 2>&1 | Out-File -Encoding utf8 -Append $log
$body = Join-Path $root "data\report_q10.md"
if (-not (Test-Path $body)) { Say "no report body"; exit 1 }

$subjectFile = Join-Path $root "data\report_q10_subject.txt"
$subject = if (Test-Path $subjectFile) { (Get-Content $subjectFile -Raw -Encoding utf8).Trim() } else { "Q10 status" }

# 2. mail it
python "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py" send `
    --model "DeepSeek V4.1 Flash Free" --project "kron_llm" `
    --subject $subject --body-file $body 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("mail exit " + $LASTEXITCODE + " | " + $subject)

# 3. marker so a scheduler does not send the same thing twice in a row
(Get-Date).ToString("o") | Out-File -Encoding utf8 (Join-Path $root "data\last_mail.txt")
