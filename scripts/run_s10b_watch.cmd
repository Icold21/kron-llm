@echo off
rem Watch the S10b run that is already queued.  Separate from the push step:
rem re-running the push would create version 2 and throw away the queued session.
rem
rem The kernel slug is the one Kaggle derived from the title
rem (kron-s10b-second-sft-pass); the metadata id said kron-s10b-sft-pass2, so the
rem first watcher polled a kernel that does not exist.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-s10b-second-sft-pass --session 17 --start-step 4521 --total-steps 14558 --push-dir kaggle/s10b --chain 1 --hours 16 --no-push --ckpt-slug icold21/kron-s10-ckpt --ckpt-title "Kron S10 SFT checkpoint" >> data\finish_session17.stdout.log 2>&1
echo [s10b watch ended with %ERRORLEVEL%] >> data\finish_session17.stdout.log
