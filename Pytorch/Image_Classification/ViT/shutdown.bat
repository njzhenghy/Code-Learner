@echo off
wmic process where "CommandLine like '%%pythonw  app.py%%' and name='pythonw.exe'" get processid,commandline >pid.txt
for /f "tokens=2 delims===" %%i in ('type pid.txt^| findstr "ProcessId="') do (taskkill /f /t /pid %%i)
pause