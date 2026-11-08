@echo off
rem Finish the decay session that is meant to end at midnight: download the
rem checkpoint, publish it and mail the report.  NO new session is pushed
rem (--no-push) -- the owner asked for a session until midnight and nothing
rem after it, so this watch stops as soon as the report is out.
rem One-shot at 00:05 rather than the 15-minute repeat: with no push there is
rem nothing to recover, and a repeating run would re-download 115 MB each time.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 13 --start-step 88936 --total-steps 109507 --chain 1 --no-push --hours 8 >> data\finish_session13.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session13.stdout.log
