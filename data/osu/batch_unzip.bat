@echo off
setlocal

for %%F in (*.zip) do (
    powershell -NoProfile -Command "Expand-Archive -LiteralPath '%%~fF' -DestinationPath '%%~dpnF' -Force"
)

echo Done.
pause
