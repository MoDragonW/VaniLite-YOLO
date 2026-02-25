# 简化版上传脚本
$gitPath = "D:\Program Files\Git\bin\git.exe"

# 执行Git命令
& $gitPath add .
& $gitPath commit -m "Initial commit"
& $gitPath checkout -b main
& $gitPath push -u origin main

if ($LASTEXITCODE -ne 0) {
    & $gitPath push -u origin main --force
}
