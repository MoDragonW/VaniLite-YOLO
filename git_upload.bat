@echo off

REM 设置Git路径
set GIT_PATH="D:\Program Files\Git\bin\git.exe"

REM 检查Git是否存在
if not exist %GIT_PATH% (
    echo Git not found at %GIT_PATH%
    echo Please update the GIT_PATH in this script
    pause
    exit /b 1
)

REM 显示Git版本
%GIT_PATH% --version

REM 初始化Git仓库
%GIT_PATH% init

REM 配置Git用户信息
%GIT_PATH% config user.name "MoDragonW"
%GIT_PATH% config user.email "syl2149335429@163.com"

REM 配置Git LFS
%GIT_PATH% lfs install

REM 添加大文件跟踪
%GIT_PATH% lfs track "*.pt"
%GIT_PATH% lfs track "*.pth"
%GIT_PATH% lfs track "*.onnx"
%GIT_PATH% lfs track "*.safetensors"
%GIT_PATH% lfs track "*.zip"
%GIT_PATH% lfs track "*.rar"
%GIT_PATH% lfs track "*.7z"

REM 添加远程仓库
%GIT_PATH% remote add origin https://github.com/MoDragonW/VaniLite-YOLO.git

REM 添加所有文件
%GIT_PATH% add .

REM 提交更改
%GIT_PATH% commit -m "Initial commit: VaniLite-YOLO project with all files"

REM 推送到GitHub
%GIT_PATH% push -u origin main

REM 如果推送失败，尝试强制推送
if %errorlevel% neq 0 (
    echo First push failed, trying force push...
    %GIT_PATH% push -u origin main --force
)

echo Done!
pause
