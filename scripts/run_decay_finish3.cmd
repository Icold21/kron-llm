@echo off
rem Finish the decay pass and hand off to S10.
rem
rem Session 15 runs the remaining ~10,600 steps (MINUTES 480; the trainer exits
rem as soon as it reaches step 109,507, so the budget is a cap, not a target).
rem When it does, this watch publishes the checkpoint, mails the report and
rem pushes kaggle/s10 with 180 minutes -- the SFT kernel guards on step >=
rem 108,507, so a short session cannot fine-tune the wrong weights.
rem
rem --chain 2: if the session is cut short (quota, platform), a second one is
rem pushed and watched; the loop stops there either way.  The fallback is 40
rem minutes on purpose: the quota on 2026-11-10 leaves ~3.6 h after this session,
rem S10 needs 2.8 h of it, so a 240-minute fallback would starve S10 and die on
rem the quota wall.  Forty minutes still covers the ~500 steps this session can
rem be short by (7.6 h of work against an 8 h cap).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-q10a-decay-pass --push-dir kaggle/q10a --session 15 --start-step 98895 --total-steps 109507 --chain 2 --minutes-plan 480,40 --hours 12 --then-push-dir kaggle/s10 --then-minutes 205 >> data\finish_session15.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session15.stdout.log
