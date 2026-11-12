@echo off
rem One battery on both checkpoints, sequentially: two evaluations at once OOM
rem the 6 GB card, and the comparison is the point -- SFT against the weights it
rem started from (the decay pass at step 109,507).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Icold\Desktop\kron_llm"
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\bench_current.py --ckpt E:/kron_out/s10_ckpt/last.pt --n 200 --blocks 512 --out data/bench_s10.json > data\bench_s10.log 2>&1
"C:\Users\Icold\AppData\Local\Programs\Python\Python311\python.exe" -u scripts\bench_current.py --ckpt E:/kron_out/q10_ckpt/last.pt --n 200 --blocks 512 --out data/bench_q10.json > data\bench_q10.log 2>&1
echo done > data\bench_done.flag
