@echo off
rem MCC VMC correction server (supervisor -> vmc_bridge). Config: AI_Streaming\server_config.json
rem Autostart at logon: see USAGE.md section 9 (schtasks).
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
"C:\Users\Contemplator\AppData\Local\Programs\Python\Python311\python.exe" AI_Streaming\server.py %*
