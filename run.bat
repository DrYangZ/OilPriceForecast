@echo off
setlocal
chcp 65001

:: 更改当前工作目录
cd .\scripts

:: 设置Python解释器的路径
set PYTHON_EXE=".\python310\python.exe"

:: 设置Python脚本的路径
set SCRIPT_PATH=".\GUI.py"

:: 使用指定的Python解释器运行脚本
start "" /wait /b cmd.exe /c
"%PYTHON_EXE%" "%SCRIPT_PATH%"

endlocal
pause
