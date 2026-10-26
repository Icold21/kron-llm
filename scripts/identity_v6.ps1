$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\bilingual\identity_v6.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== identity v6 $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# ---------------------------------------------------------- 1. wait for RU v2
$deadline = (Get-Date).AddHours(4)
$ruLog = Join-Path $root "data\bilingual\identity_ru_rick_v2.log"
while ($true) {
    if ((Test-Path $ruLog) -and ((Get-Content $ruLog -Raw) -match [regex]::Escape("=== done"))) { break }
    if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for RU rick v2"; exit 1 }
    Start-Sleep -Seconds 60
}
Say "RU rick v2 finished"

# ------------------------------------------------- 2. EN pastiche regeneration
Say "generating EN rick (monologue prompts)"
python -u scripts\gen_identity.py --samples-per-language 850 --seed 402 --style rick --lang en --force 2>&1 | Out-File -Encoding utf8 -Append $log
Say ("en rick exit " + $LASTEXITCODE)

# ------------------------------------------------------------- 3. repair all
$jobs = @(
    @("identity_en", 800, "plain", 601),
    @("identity_ru", 800, "plain", 602),
    @("identity_rick_en", 850, "rick", 603),
    @("identity_rick_ru", 850, "rick", 604)
)
foreach ($j in $jobs) {
    Say ("repair " + $j[0])
    python -u scripts\repair_identity.py --name $j[0] --target $j[1] --style $j[2] --seed $j[3] 2>&1 | Out-File -Encoding utf8 -Append $log
    Say ("repair exit " + $LASTEXITCODE)
}

# ----------------------------------------------------------------- 4. audit
python -u scripts\audit_identity.py 2>&1 | Out-File -Encoding utf8 -Append $log
Say "audit written"
Say "IDENTITY V6 DONE"
