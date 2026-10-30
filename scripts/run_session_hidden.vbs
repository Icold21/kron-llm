' Launch the 03:05 session guard with no console window.
'
' Same reasoning as run_chain_hidden.vbs: wscript.exe has no console, so the
' guard and the chain it starts never flash a window that invites being closed.
'
' Called by the scheduled task `kron_session_0300`; not meant to be run by hand.

Dim sh, root
root = "C:\Users\Icold\Desktop\kron_llm"
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & root & "\scripts\session_guard.ps1""", 0, False
