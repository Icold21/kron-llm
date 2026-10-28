$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Icold\Desktop\kron_llm"
Set-Location $root
$log = Join-Path $root "data\q10_chain.log"
$kernel = "icold21/kron-q10-10m-bilingual-pretraining"
$code = "icold21/kron-llm-code-v15"
$mail = "$env:USERPROFILE\.config\opencode\skills\email-control\scripts\mail.py"
# The mailer refuses to send without a session label (the client update made it
# mandatory).  The chain has no OpenCode session of its own, so it labels its
# reports with its own name.
$label = "kron_chain"

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
        python $mail send --model "DeepSeek V4.1 Flash" --project "kron_llm" --session $label `
            --subject $subject --body-file $htmlPath --html 2>&1 |
            Out-File -Encoding utf8 -Append $log
    } elseif (Test-Path $mdPath) {
        python $mail send --model "DeepSeek V4.1 Flash" --project "kron_llm" --session $label `
            --subject $subject --body-file $mdPath 2>&1 |
            Out-File -Encoding utf8 -Append $log
    }
}

# Session reports are held until 08:00: a mail that arrives at 03:00 is read
# half-asleep or not at all, and the point of the report is that someone reads
# it.  The chain cannot sleep to achieve that -- it would stop watching the next
# session -- so the queue is a marker file plus a one-off task `kron_report_mail`
# that runs scripts/send_pending_report.ps1 at 08:00.  A report generated after
# 08:00 is sent at once.  Failures never go through here: an error mail must not
# sit for five hours.
function QueueReport($subject, $htmlPath) {
    $due = (Get-Date).Date.AddHours(8)
    if ((Get-Date) -ge $due) {
        SendReport $subject $null $htmlPath
        return
    }
    $file = Join-Path $root "data\report_mail_pending.json"
    # ConvertFrom-Json hands back a JSON array as one object, so wrapping it in
    # @() would nest the whole queue as a single entry; unwrap it by hand.
    $queue = @()
    if (Test-Path $file) {
        $parsed = Get-Content $file -Raw -Encoding utf8 | ConvertFrom-Json
        if ($parsed -is [System.Array]) { $queue = $parsed } elseif ($null -ne $parsed) { $queue = @($parsed) }
    }
    $queue += @{ subject = $subject
                 html = $htmlPath.Substring($root.Length + 1)
                 queued = (Get-Date -Format "s") }
    ConvertTo-Json -InputObject @($queue) -Depth 3 | Out-File -Encoding utf8 $file

    # /SC ONCE without /SD defaults to today, so this needs no locale-specific
    # date and cannot fire on the wrong day.
    $action = 'wscript.exe //B "C:\Users\Icold\Desktop\kron_llm\scripts\run_report_mail_hidden.vbs"'
    schtasks /Create /TN "kron_report_mail" /TR $action /SC ONCE /ST 08:00 /F 2>&1 |
        Out-File -Encoding utf8 -Append $log
    Say ("report queued for 08:00 (" + $queue.Count + " in the queue): " + $subject)
}

# External calls can hang: the Kaggle CLI has no timeout of its own, and a stuck
# request once blocked this chain for 20 minutes while the run kept burning GPU
# hours.  Every API call goes through a job with a deadline; a timeout is logged
# and treated as "no information", never as a result.
function Run($timeoutSec, [string[]]$pyArgs) {
    # Start-Job runs in the user's home directory, not the repo, so every
    # relative path inside the job ("kaggle/q10", "scripts\...") silently missed
    # when the chain was started by Task Scheduler instead of from the repo.
    $job = Start-Job -ScriptBlock { param($a, $wd) Set-Location $wd; & python @a 2>&1 } `
        -ArgumentList (,$pyArgs), $root
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
    # A push can be refused because the weekly GPU quota is spent.  That is a
    # calendar event, not a failure, so it is retried every three hours until it
    # goes through -- the alternative was the chain seeing the previous version's
    # COMPLETE, downloading the same checkpoint, deciding "no progress" and
    # mailing a false failure every five minutes.
    while ($true) {
        $push = Run 600 @('-m','kaggle','kernels','push','-p','kaggle/q10')
        $push | Out-File -Encoding utf8 -Append $log
        $line = (($push | Select-Object -Last 1) -join " ").Trim()
        Say ("push: " + $line)
        if ($line -match "quota|Quota|QUOTA") {
            Say "GPU QUOTA WALL: retrying the push in 3 hours"
            Start-Sleep -Seconds 10800
            continue
        }
        if ($line -match "successfully|Kernel version") { return "ok" }
        return "error"
    }
}

"=== chain started $(Get-Date -Format 'HH:mm:ss')" | Out-File -Encoding utf8 $log

$prevStep = [int](python scripts\session_progress.py --history 2>$null | Select-Object -Last 1)
Say ("last step seen before this session: " + $prevStep)
# `kernels status` can report the *previous* version's state, so a COMPLETE that
# arrives within minutes of a push is stale, not the end of the new session.
$pushedAt = (Get-Date).AddHours(-2)
$stalled = 0

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
        if ($status -match "COMPLETE|ERROR|CANCEL|FAILED") {
            if ((Get-Date) -lt $pushedAt.AddMinutes(10)) {
                Say "status is the previous version's; waiting for the new run to start"
                Start-Sleep -Seconds 300
                continue
            }
            break
        }
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

    # 3. no progress: the run either died or never started at all (a refused push
    #    and a spent GPU quota both look like this).  Pushing again is the right
    #    response to both, but a real crash would then loop forever, so three
    #    consecutive stalls are reported as a failure -- and only the third.
    if ($nowStep -le $prevStep) {
        Say "no progress this session, pushing again"
        # PushSession sleeps through a spent weekly quota on its own, so waiting
        # for the reset never counts as a failed attempt.
        if ((PushSession) -ne "ok") { Say "push did not go through either" }
        $stalled++
        if ($stalled -ge 3) {
            python scripts\kaggle_log.py "" 20 4000 > data\session_failure.txt 2>&1
            $tail = (Get-Content "data\session_failure.txt" -Tail 60 -Encoding utf8) -join "`n"
            $md = Join-Path $root "data\session_failure.md"
            $head = "# Q10: three sessions in a row without progress`n`n"
            $head += "The kernel ended with status ``$status``, but the output still has no step beyond $prevStep.`n"
            $head += "Each retry pushed the same session again; three of them came back empty, so this`n"
            $head += "is not a quota wait -- something is wrong with the run and it needs a look.`n`n"
            $head += "## Log tail`n`n``````n" + $tail + "`n``````"
            $head | Out-File -Encoding utf8 $md
            SendReport "Q10: three sessions in a row failed without progress" $md (Join-Path $root "data\session_failure.html")
            Say "FAILED SESSION: reported after $stalled stalls, stopping"
            exit 3
        }
        Say "retry $stalled of 3 queued"
        $pushedAt = Get-Date
        Start-Sleep -Seconds 120
        continue session
    }
    $stalled = 0

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
    if ($errored) { SendReport $subject $md $html } else { QueueReport $subject $html }

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
    if ((PushSession) -ne "ok") { Say "push did not go through" }
    $pushedAt = Get-Date
    Say "NEXT SESSION PUSHED"
    Start-Sleep -Seconds 180
}
