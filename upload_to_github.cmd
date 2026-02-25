@echo off

REM 简化版Git上传脚本
REM 请在命令提示符(cmd.exe)中运行，不要在PowerShell中运行

REM 设置Git路径
set "GIT_PATH=D:\Program Files\Git\bin\git.exe"

REM 检查Git是否存在
if not exist "%GIT_PATH%" (
    echo Git not found at %GIT_PATH%
    echo Please update the GIT_PATH in this script
    pause
    exit /b 1
)

REM 显示Git版本
"%GIT_PATH%" --version

REM 初始化Git仓库
"%GIT_PATH%" init

REM 配置Git用户信息
"%GIT_PATH%" config user.name "MoDragonW"
"%GIT_PATH%" config user.email "syl2149335429@163.com"

REM 配置Git LFS
"%GIT_PATH%" lfs install

REM 添加大文件跟踪
"%GIT_PATH%" lfs track "*.pt"
"%GIT_PATH%" lfs track "*.pth"
"%GIT_PATH%" lfs track "*.onnx"
"%GIT_PATH%" lfs track "*.safetensors"
"%GIT_PATH%" lfs track "*.zip"
"%GIT_PATH%" lfs track "*.rar"
"%GIT_PATH%" lfs track "*.7z"

REM 使用令牌添加远程仓库
set "GITHUB_TOKEN=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"
set "REMOTE_URL=https://%GITHUB_TOKEN%@github.com/MoDragonW/VaniLite-YOLO.git"

"%GIT_PATH%" remote add origin "%REMOTE_URL%"

REM 添加所有文件
"%GIT_PATH%" add .

REM 提交更改
"%GIT_PATH%" commit -m "Initial commit: VaniLite-YOLO project with all files"

REM 创建并切换到main分支
"%GIT_PATH%" checkout -b main

REM 推送到GitHub
echo Pushing to GitHub...
"%GIT_PATH%" push -u origin main

REM 如果推送失败，尝试强制推送
if %errorlevel% neq 0 (
    echo First push failed, trying force push...
    "%GIT_PATH%" push -u origin main --force
)

REM 检查结果
if %errorlevel% equ 0 (
    echo ✅ Successfully pushed to GitHub!
    echo Repository: https://github.com/MoDragonW/VaniLite-YOLO
) else (
    echo ❌ Push failed. Please check:
    echo 1. GitHub repository exists
    echo 2. Network connection is stable
    echo 3. Token has correct permissions
)

echo Done!
pause
