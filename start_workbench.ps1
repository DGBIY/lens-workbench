# 镜头设计工作台 - 一键启动脚本（本地 + 分享双兼容）
# 由《启动工作台.bat》调用；也可以右键"使用 PowerShell 运行"
$ErrorActionPreference = 'Stop'

# ---- 本地 GA 环境（路径存在时自动启用；对方机器没有该路径则自动跳过）----
if (Test-Path "C:\Users\Administrator\Desktop\GA\完整项目") {
    $env:GA_PROJECT_ROOT = "C:\Users\Administrator\Desktop\GA\完整项目"
}

# 进入工作台目录（脚本所在目录）
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host ""
Write-Host "  正在启动镜头设计工作台..."
Write-Host "  浏览器将自动打开 http://localhost:8501"
Write-Host "  关闭本窗口 = 停止工作台"
Write-Host ""

# ---- Python 解释器：优先本机 python_env；对方机器自动用 PATH 里的 python ----
$python = "C:\Users\Administrator\Desktop\GA\完整项目\python_env\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
    Write-Host "  [分享模式] 使用系统 Python（需已安装 requirements.txt 依赖）"
}

& $python -m streamlit run app.py