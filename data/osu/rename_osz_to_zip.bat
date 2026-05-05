@echo off
setlocal enabledelayedexpansion

for %%F in (*.osz) do (
    ren "%%F" "%%~nF.zip"
)

echo Done.
pause
