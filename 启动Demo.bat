@echo off
chcp 65001 >nul
title 低空企业信贷风险评估
echo ==============================================
echo   正在启动信贷风险评估 Demo...
echo   启动成功后浏览器会自动打开
echo ==============================================
echo.

cd /d "%~dp0"

pip install -r requirements.txt

echo.
echo 启动中...
streamlit run app.py

pause