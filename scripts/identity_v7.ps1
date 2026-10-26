$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\bilingual\identity_v7.log"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

"=== identity v7 $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

# The pastiche is generated at temperature 0.7: at 0.95 the 4B model produced
# invented creatures and broken grammar (one in three texts), at 0.7 eleven of
# twelve samples pass the fact gate and read as prose.

Say "EN pastiche: 850 at temp 0.7"
python -u scripts\gen_identity.py --samples-per-language 850 --seed 402 `
    --style rick --lang en --temperature 0.7 --force 2>&1 |
    Out-File -Encoding utf8 -Append $log
Say ("en rick exit " + $LASTEXITCODE)

Say "RU pastiche: 850 at temp 0.7"
python -u scripts\gen_identity.py --samples-per-language 850 --seed 405 `
    --style rick --lang ru --temperature 0.7 --force 2>&1 |
    Out-File -Encoding utf8 -Append $log
Say ("ru rick exit " + $LASTEXITCODE)

$jobs = @(
    @("identity_en", 800, "plain", 601),
    @("identity_ru", 800, "plain", 602),
    @("identity_rick_en", 850, "rick", 603),
    @("identity_rick_ru", 850, "rick", 604)
)
foreach ($j in $jobs) {
    Say ("repair " + $j[0])
    python -u scripts\repair_identity.py --name $j[0] --target $j[1] `
        --style $j[2] --seed $j[3] --temperature 0.7 2>&1 |
        Out-File -Encoding utf8 -Append $log
    Say ("repair exit " + $LASTEXITCODE)
}

python -u scripts\audit_identity.py 2>&1 | Out-File -Encoding utf8 -Append $log
Say "audit written"
Say "IDENTITY V7 DONE"
