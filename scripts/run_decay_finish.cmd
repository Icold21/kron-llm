@echo off
rem Finish the decay session: download the checkpoint, publish it and mail the
rem report.  NO new session is pushed (--no-push).
rem
rem The flag file at the end is what the shutdown task waits for: the machine must
rem not go down in the middle of an upload, and the session is still running when
rem this fires, so the wait is unbounded from here and bounded there.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
if exist "data\finish_done.flag" del "data\finish_done.flag"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 13 --start-step 88936 --total-steps 109507 --chain 1 --no-push --hours 8 >> data\finish_session13.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session13.stdout.log
echo done > "data\finish_done.flag"
