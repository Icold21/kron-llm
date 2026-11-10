@echo off
rem Finish the S10 fine-tune: download the checkpoint, publish it as a new
rem version of kron-s10-ckpt, generate samples and mail the report.
rem
rem --no-push is deliberate: S10b costs 6.3 h, which the week's quota does not
rem have, and the decision belongs to the verdict (scripts/verdict_s10.py), not
rem to a timer.  --ckpt-slug keeps the fine-tuned weights out of kron-q10-ckpt,
rem which must stay the pretraining checkpoint.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\finish_session.py --kernel icold21/kron-s10-supervised-fine-tuning --session 16 --start-step 0 --total-steps 4800 --chain 1 --no-push --hours 8 --out E:\kron_out\session16 --ckpt-slug icold21/kron-s10-ckpt --ckpt-title "Kron S10 SFT checkpoint" >> data\finish_session16.stdout.log 2>&1
echo [watch ended with %ERRORLEVEL%] >> data\finish_session16.stdout.log
