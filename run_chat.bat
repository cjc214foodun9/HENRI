@echo off
echo =====================================================================
echo              BOOTSTRAPPING HENRI COGNITIVE CHATBOT SERVER             
echo =====================================================================
echo [*] Launching your web browser at http://127.0.0.1:5000...
start "" "http://127.0.0.1:5000"
echo [*] Starting Python backend server...
C:\Python312\python.exe chat_server.py
pause
