@echo off
rem Watch the S10b run that started at 12:24 on 2026-11-15 and must end by 22:30.
rem
rem The kernel computes its own budget from a 19:30 UTC deadline, so the stop time
rem holds whatever the queue does; this watcher only publishes and reports.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-s10b-second-sft-pass --session 19 --start-step 4521 --total-steps 18000 --push-dir kaggle/s10b --chain 1 --hours 12 --no-push --ckpt-slug icold21/kron-s10-ckpt --ckpt-title "Kron S10 SFT checkpoint" >> data\finish_session19.stdout.log 2>&1
echo [s10b session 19 watch ended with %ERRORLEVEL%] >> data\finish_session19.stdout.log
