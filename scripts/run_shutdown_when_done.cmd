@echo off
rem Switch the machine off, but only once the midnight session is published.
rem
rem The finish task (kron_decay_finish) starts waiting at 00:05 and the session
rem may still be running; it writes data\finish_done.flag only after the download,
rem the dataset version and the report are done.  This waits for that flag, and
rem gives up after three hours regardless -- after that the checkpoint is still in
rem the kernel output on Kaggle and can be fetched next time, so keeping the box
rem awake buys nothing.
set FLAG=C:\Users\Icold\Desktop\kron_llm\data\finish_done.flag
set LOG=C:\Users\Icold\Desktop\kron_llm\data\shutdown.log
echo [%DATE% %TIME%] waiting for %FLAG% >> "%LOG%"
for /L %%i in (1,1,90) do (
    if exist "%FLAG%" goto :shutdown
    timeout /t 120 /nobreak >nul
)
echo [%DATE% %TIME%] flag never appeared, shutting down anyway >> "%LOG%"
:shutdown
echo [%DATE% %TIME%] shutting down >> "%LOG%"
shutdown /s /t 120 /c "KronLLM: session published, shutting down"
