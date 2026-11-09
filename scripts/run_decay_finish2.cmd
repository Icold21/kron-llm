@echo off
rem Finish the 2-hour decay session: download the checkpoint, publish it and mail
rem the report.  NO new session is pushed (--no-push).
rem
rem Safe to run repeatedly: finish_session keeps data/finish_state.json, so a
rem restart skips the upload for a step that is already published and stops a
rem second push for a step that was already pushed.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 14 --start-step 96120 --total-steps 109507 --chain 1 --no-push --hours 6 >> data\finish_session14.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session14.stdout.log
