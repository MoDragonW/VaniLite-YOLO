# PowerShell脚本：上传VaniLite-YOLO项目到GitHub

# 设置Git路径
$gitPath = "D:\Program Files\Git\bin\git.exe"

# 显示Git版本
& $gitPath --version

# 配置用户信息
& $gitPath config user.name "MoDragonW"
& $gitPath config user.email "syl2149335429@163.com"

# 配置Git LFS
& $gitPath lfs install

# 添加大文件跟踪
& $gitPath lfs track "*.pt"
& $gitPath lfs track "*.pth"
& $gitPath lfs track "*.onnx"
& $gitPath lfs track "*.safetensors"
& $gitPath lfs track "*.zip"
& $gitPath lfs track "*.rar"
& $gitPath lfs track "*.7z"

# 添加远程仓库
$token = "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"
$remoteUrl = "https://$token@github.com/MoDragonW/VaniLite-YOLO.git"
& $gitPath remote add origin $remoteUrl

# 添加所有文件
& $gitPath add .

# 提交更改
& $gitPath commit -m "Initial commit"

# 创建并切换到main分支
& $gitPath checkout -b main

# 推送到GitHub
Write-Host "Pushing to GitHub..."
& $gitPath push -u origin main

# 检查结果
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully pushed to GitHub!"
    Write-Host "Repository: https://github.com/MoDragonW/VaniLite-YOLO"
} else {
    Write-Host "❌ First push failed, trying force push..."
    & $gitPath push -u origin main --force
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully pushed to GitHub!"
        Write-Host "Repository: https://github.com/MoDragonW/VaniLite-YOLO"
    } else {
        Write-Host "❌ Push failed."
    }
}

Write-Host "Done!"
