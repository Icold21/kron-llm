' Launch the training chain with no console window.
'
' Task Scheduler starts powershell.exe in the user's session, so even with
' -WindowStyle Hidden a console window flashes and sometimes stays open -- which
' looks exactly like a runaway process and invites closing it, killing the
' watcher of a 500-minute GPU session.  wscript.exe has no console of its own, so
' running the chain through it produces no window at all.
'
' Called by the scheduled task `kron_chain`; not meant to be run by hand.

Dim sh, root
root = "C:\Users\Icold\Desktop\kron_llm"
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & root & "\scripts\chain_q10.ps1""", 0, False
