@echo off
rem Watch the Q10A decay pass to its end (step 109,507), then push S10.
rem Launched by Task Scheduler, so it must survive without a console: UTF-8 is
rem forced here and the transcript is appended to a file instead of a window.
rem
rem --end-by 10:00,,,  the session pushed after session 11 must end by 10:00,
rem                    so its budget is computed from the clock at push time
rem                    (queue + bin copy + teardown reserved).  The later
rem                    sessions fall back to --minutes-plan: 680 for the one
rem                    that is expected to finish the decay (680 + 13 min of
rem                    overhead still beats the 720-minute platform cap), 300
rem                    for whatever is left.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 11 --start-step 73636 --total-steps 109507 --chain 5 --minutes-plan 190,680,300,300 --end-by 10:00,,, --hours 13 --then-push-dir kaggle/s10 --then-minutes 180 >> data\finish_session11.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session11.stdout.log
