@echo off
rem Watch the Q10A decay pass to its end (step 109,507), then push S10.
rem Launched by Task Scheduler, so it must survive without a console: UTF-8 is
rem forced here and the transcript is appended to a file instead of a window.
rem --chain 3 with 660,660,260 minutes: two full sessions plus the tail, which
rem is sized so the decay ends inside the weekly 30 h of GPU quota and the S10
rem push still happens on the same allowance.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 11 --start-step 73636 --total-steps 109507 --chain 3 --minutes-plan 660,660,260 --hours 13 --then-push-dir kaggle/s10 --then-minutes 180 >> data\finish_session11.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session11.stdout.log
