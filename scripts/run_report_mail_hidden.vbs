' Launch the queued-report mailer with no console window.
'
' Same trick as run_chain_hidden.vbs: Task Scheduler starts powershell.exe inside
' the user's desktop session, so a console window flashes even with -WindowStyle
' Hidden.  wscript.exe has no console of its own, so the 08:00 mail job is silent.

Dim sh, root
root = "C:\Users\Icold\Desktop\kron_llm"
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & root & "\scripts\send_pending_report.ps1""", 0, False
