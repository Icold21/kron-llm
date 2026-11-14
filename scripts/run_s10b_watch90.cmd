@echo off
rem Watch the 90-minute S10b session (version 3 of the kernel).
rem
rem The 22:30-deadline version refused to start when its slot finally came at
rem 22:25, which is exactly what it was built to do -- but it burned the slot.
rem This one carries a plain 90-minute budget: it trains for 90 minutes from
rem whenever it starts, and the LR anneal is sized to that.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-s10b-second-sft-pass --session 18 --start-step 4521 --total-steps 6506 --push-dir kaggle/s10b --chain 1 --hours 8 --no-push --ckpt-slug icold21/kron-s10-ckpt --ckpt-title "Kron S10 SFT checkpoint" >> data\finish_session18.stdout.log 2>&1
echo [s10b 90min watch ended with %ERRORLEVEL%] >> data\finish_session18.stdout.log
