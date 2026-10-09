@echo off
rem MCC VMC correction server (supervisor -> vmc_bridge). Config: AI_Streaming\server_config.json
rem Autostart at logon: see USAGE.md section 9 (schtasks).
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
rem Uses "python" on PATH. To pin a specific interpreter, replace it with the full path to python.exe.
python AI_Streaming\server.py %*
