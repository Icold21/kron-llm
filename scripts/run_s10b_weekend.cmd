@echo off
rem S10b, first session: push the kernel and watch it to completion.
rem
rem Fires at 03:05 MSK, five minutes after the weekly GPU quota resets (00:00
rem UTC).  One session is one complete anneal -- 14,558 steps, 238M tokens of the
rem mixed corpus -- and the watch publishes the checkpoint and mails the report.
rem
rem --no-push is deliberate: whether a second session is worth the remaining
rem hours depends on this run's numbers, and a resume kernel (not another
rem weights-only start) is what would continue it.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u -m kaggle kernels push -p kaggle/s10b >> data\s10b_push.log 2>&1
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-s10b-sft-pass2 --session 17 --start-step 4521 --total-steps 14558 --push-dir kaggle/s10b --chain 1 --hours 16 --no-push --ckpt-slug icold21/kron-s10-ckpt --ckpt-title "Kron S10 SFT checkpoint" >> data\finish_session17.stdout.log 2>&1
echo [s10b session 1 ended with %ERRORLEVEL%] >> data\finish_session17.stdout.log
