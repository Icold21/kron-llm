$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_chain.log"
$kernel = "icold21/kron-q10-10m-bilingual-pretraining"
$code = "icold21/kron-llm-code-v15"
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    $line | Out-File -Encoding utf8 -Append $log
}

# Every report goes out as HTML: Markdown pipe tables do not render in mail
# clients, and the tables are the point.  Markdown stays as the repo record.
function SendReport($subject, $mdPath, $htmlPath) {
    if (-not (Test-Path $htmlPath) -and (Test-Path $mdPath)) {
        python scripts\md_to_html.py $mdPath $htmlPath 2>&1 |
            Out-File -Encoding utf8 -Append $log
    }
    if (Test-Path $htmlPath) {
        python $mail send --model "DeepSeek V4.1 Flash Free" --project "kron_llm" `
            --subject $subject --body-file $htmlPath --html 2>&1 |
            Out-File -Encoding utf8 -Append $log
    } elseif (Test-Path $mdPath) {
        python $mail send --model "DeepSeek V4.1 Flash Free" --project "kron_llm" `
            --subject $subject --body-file $mdPath 2>&1 |
            Out-File -Encoding utf8 -Append $log
    }
}

# External calls can hang: the Kaggle CLI has no timeout of its own, and a stuck
# request once blocked this chain for 20 minutes while the run kept burning GPU
# hours.  Every API call goes through a job with a deadline; a timeout is logged
# and treated as "no information", never as a result.
function Run($timeoutSec, [string[]]$pyArgs) {
    $job = Start-Job -ScriptBlock { param($a) & python @a 2>&1 } -ArgumentList (,$pyArgs)
    if (Wait-Job $job -Timeout $timeoutSec) {
        $result = Receive-Job $job
    } else {
        Stop-Job $job -ErrorAction SilentlyContinue
        $result = @("TIMEOUT after $timeoutSec s: python $($pyArgs -join ' ')")
    }
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    return $result
}

function PushSession() {
    $push = Run 600 @('-m','kaggle','kernels','push','-p','kaggle/q10')
    $push | Out-File -Encoding utf8 -Append $log
    Say ("push: " + (($push | Select-Object -Last 1) -join " "))
}

"=== chain started $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

$prevStep = [int](python scripts\session_progress.py --history 2>$null | Select-Object -Last 1)
Say ("last step seen before this session: " + $prevStep)

# One instance now drives the whole run: wait -> download -> report -> publish ->
# push -> wait again.  It used to handle a single session and exit, which meant
# nothing was watching the next one; and a GPU-quota wall must not look like a
# failed session, or the run stops until someone wakes up.
:session while ($true) {

    # 1. wait for the session to end.  A transient API failure must not look like
    #    completion, otherwise the next session would be pushed on top of a live run.
    $deadline = (Get-Date).AddHours(14)
    while ($true) {
        if ((Get-Date) -gt $deadline) { Say "TIMEOUT waiting for the session"; exit 1 }
        $status = Run 180 @('-m','kaggle','kernels','status',$kernel) | Select-Object -Last 1
        Say $status
        if ($status -match "COMPLETE|ERROR|CANCEL|FAILED") { break }
        Start-Sleep -Seconds 300
    }
    $errored = $status -match "ERROR|CANCEL|FAILED"
    Say "session ended, being processed"

    # 2. download the session output.  The directory is cleared first: a failed run
    #    can hand back the previous version's files, and stale weights are worse
    #    than no weights.
    $out = Join-Path $root "data\kaggle_out"
    Remove-Item (Join-Path $out "q10") -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    Run 1800 @('-m','kaggle','kernels','output',$kernel,'-p',$out) |
        Out-File -Encoding utf8 -Append $log
    Say "output download finished"

    $nowStep = [int](python scripts\session_progress.py --new 2>$null | Select-Object -Last 1)
    Say ("this download ends at step " + $nowStep)

    # 3. no progress: either the run died, or Kaggle refused to start it because
    #    the weekly GPU quota is spent.  The first is a bug, the second is a
    #    calendar event -- waiting for it costs nothing but a sleep.
    if ($nowStep -le $prevStep) {
        $sessionLog = Join-Path $out "kron-q10-10m-bilingual-pretraining.log"
        $text = if (Test-Path $sessionLog) { Get-Content $sessionLog -Raw -Encoding utf8 } else { "" }
        if ($text -match "quota|Quota|QUOTA") {
            Say "GPU QUOTA WALL: waiting 3 hours before pushing again"
            Start-Sleep -Seconds 10800
            PushSession
            continue session
        }
        python scripts\kaggle_log.py "" 20 4000 > data\session_failure.txt 2>&1
        $tail = (Get-Content "data\session_failure.txt" -Tail 60 -Encoding utf8) -join "`n"
        $md = Join-Path $root "data\session_failure.md"
        $head = "# Q10: session failed, no progress`n`n"
        $head += "The kernel ended with status ``$status``, but the output has no step beyond $prevStep.`n"
        $head += "No new checkpoint was written, so nothing is published and nothing is resumed.`n`n"
        $head += "## Log tail`n`n``````n" + $tail + "`n``````"
        $head | Out-File -Encoding utf8 $md
        SendReport "Q10: session failed without progress" $md (Join-Path $root "data\session_failure.html")
        Say "FAILED SESSION: reported, nothing pushed"
        exit 3
    }

    # 3b. honest validation: score the downloaded checkpoint on a strided sample of
    #     the whole val bin, and once with GPT-2 on the same blocks.  The stock
    #     val_bpb only sees the first 262k tokens of the bin, which are English code.
    $ckptNow = Join-Path $root "data\kaggle_out\q10\last.pt"
    if (Test-Path $ckptNow) {
        Say "scoring the checkpoint on the strided val sample"
        Run 1800 @('scripts\eval_reference.py','--source','ckpt','--ckpt',$ckptNow,
                   '--blocks','1024','--out','data\baseline_latest.json') |
            Out-File -Encoding utf8 -Append $log
        if (-not (Test-Path "data\baseline_gpt2.json")) {
            Say "no GPT-2 baseline yet, computing one on the same blocks"
            Run 3600 @('scripts\eval_reference.py','--source','gpt2','--blocks','1024',
                       '--out','data\baseline_gpt2.json') |
                Out-File -Encoding utf8 -Append $log
        }
        # Generation goes in the report: the bpb table says whether the loss
        # moved, the samples say whether the model became a better writer.
        Run 900 @('scripts\report_samples.py','--ckpt',$ckptNow,
                  '--out','data\report_samples.md') |
            Out-File -Encoding utf8 -Append $log
        Say "generation samples refreshed"
    }

    # 4. report the curve and mail it
    Remove-Item "data\q10_epoch_done.flag" -ErrorAction SilentlyContinue
    python scripts\report_q10.py --tag session 2>&1 | Out-File -Encoding utf8 -Append $log
    $md = Join-Path $root "data\report_q10.md"
    $html = Join-Path $root "data\report_q10.html"
    $subjectFile = Join-Path $root "data\report_q10_subject.txt"
    $subject = if (Test-Path $subjectFile) { (Get-Content $subjectFile -Raw -Encoding utf8).Trim() } else { "Q10 session report" }
    if ($errored) { $subject = $subject + " (session ended with an error)" }
    if ($errored -and (Test-Path $html)) {
        python scripts\kaggle_log.py "" 20 3000 > data\session_failure.txt 2>&1
        $tail = (Get-Content "data\session_failure.txt" -Tail 40 -Encoding utf8) -join "`n"
        $encoded = [System.Net.WebUtility]::HtmlEncode($tail)
        $suffix = '<hr><h3>Log tail</h3><pre style="background:#f6f8fa;border:1px solid #e1e4e8;padding:10px;font-size:12px;overflow-x:auto">' + $encoded + '</pre>'
        Add-Content -Encoding utf8 $html $suffix
    }
    SendReport $subject $md $html

    # 5. publish the checkpoint so the next session can attach it
    if (-not (Test-Path $ckptNow)) {
        Say "no last.pt in the output; stopping before the resume"
        exit 2
    }
    $stage = Join-Path $root "data\kaggle_ckpt"
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item $ckptNow (Join-Path $stage "last.pt") -Force
    python -c "import json; json.dump({'title':'Kron Q10 checkpoint','id':'icold21/kron-q10-ckpt','licenses':[{'name':'other'}]}, open(r'$stage\dataset-metadata.json','w'), indent=2)"
    $ver = Run 1800 @('-m','kaggle','datasets','version','-p',$stage,'-m',"session checkpoint step $nowStep")
    $ver | Out-File -Encoding utf8 -Append $log
    if (-not ($ver -match "successfully|being created|Dataset version")) {
        Say "version failed or timed out, trying create"
        Run 1800 @('-m','kaggle','datasets','create','-p',$stage) |
            Out-File -Encoding utf8 -Append $log
    }
    Say "checkpoint dataset updated"

    # 6. the epoch is complete: keep the weights and stop; the WSD decay pass is
    #    launched separately, on curated data.
    if (Test-Path "data\q10_epoch_done.flag") {
        Say "EPOCH COMPLETE: checkpoint published, waiting for the decay pass"
        Say "NEXT: launch Q10A (decay, 29.6% of an epoch, LR 3e-4 -> 1e-6)"
        exit 0
    }

    # 7. push the next session and wait for it
    $prevStep = $nowStep
    PushSession
    Say "NEXT SESSION PUSHED"
    Start-Sleep -Seconds 180
}
