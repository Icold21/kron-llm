@echo off
rem Watch the Q10A decay pass to its end (step 109,507), then push S10.
rem Launched by Task Scheduler, so it must survive without a console: UTF-8 is
rem forced here and the transcript is appended to a file instead of a window.
rem
rem The repeating trigger re-runs this every 15 minutes so a killed watcher comes
rem back on its own.  That is only safe because finish_session keeps a per-kernel
rem state file (data/finish_state.json): a step that was already published is not
rem uploaded again, a step already pushed does not get a second session, and a
rem step retried three times without getting past the report stops the watch.
rem Without that guard the 2026-11-08 report bug re-uploaded 115 MB every 15
rem minutes for five hours.
rem
rem Session 12 (660 min) is expected to end ~22:40 around step 104,500; the 240
rem minutes after it cover the remaining ~5,000 steps.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 12 --start-step 88936 --total-steps 109507 --chain 2 --minutes-plan 240,240 --hours 13 --then-push-dir kaggle/s10 --then-minutes 180 >> data\finish_session12.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session12.stdout.log
