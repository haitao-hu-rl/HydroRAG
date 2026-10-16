# ============================================================================
# HydroRAG 一键启动
#
#   .\start.ps1                 启动服务并打开浏览器
#   .\start.ps1 -Port 8080      换端口
#   .\start.ps1 -Reload         开发模式（改动代码自动重载）
#   .\start.ps1 -NoBrowser      只启动服务，不打开浏览器
#   .\start.ps1 -Stop           停止已运行的 HydroRAG 服务
#
# 双击运行请用同目录下的 start.bat。
# ============================================================================

[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$Reload,
    [switch]$NoBrowser,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BaseUrl = "http://127.0.0.1:$Port"
$HealthUrl = "$BaseUrl/api/health"

function Write-Step($Text) { Write-Host "  $Text" -ForegroundColor Gray }
function Write-Ok($Text) { Write-Host "  $Text" -ForegroundColor Green }
function Write-Warn2($Text) { Write-Host "  $Text" -ForegroundColor Yellow }
function Write-Err($Text) { Write-Host "  $Text" -ForegroundColor Red }

function Get-HydroRagProcesses([int]$ListenPort) {
    $connections = Get-NetTCPConnection -State Listen -LocalPort $ListenPort -ErrorAction SilentlyContinue
    if (-not $connections) { return @() }
    $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    return @($pids | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue } | Where-Object { $_ })
}

function Test-ServiceReady([int]$ListenPort) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$ListenPort/api/health" -TimeoutSec 3
        return $response.ok -eq $true
    } catch {
        return $false
    }
}

Write-Host ""
Write-Host "HydroRAG 启动器" -ForegroundColor Cyan
Write-Host ("=" * 46) -ForegroundColor DarkGray

# ---------------------------------------------------------------- 停止模式 ---
if ($Stop) {
    $running = Get-HydroRagProcesses $Port
    if (-not $running) {
        Write-Warn2 "端口 $Port 上没有正在运行的服务。"
        exit 0
    }
    foreach ($proc in $running) {
        Write-Step "停止进程 $($proc.Id) ($($proc.ProcessName))"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 700
    if (Get-HydroRagProcesses $Port) { Write-Err "停止失败，请手动结束进程。" ; exit 1 }
    Write-Ok "服务已停止。"
    exit 0
}

# ------------------------------------------------------------ 1. 环境检查 ---
Write-Host "[1/4] 检查运行环境" -ForegroundColor White

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Err "找不到虚拟环境：$VenvPython"
    Write-Host ""
    Write-Host "  请先在项目目录创建虚拟环境并安装依赖：" -ForegroundColor Yellow
    Write-Host "    cd `"$ProjectRoot`"" -ForegroundColor Gray
    Write-Host "    python -m venv .venv" -ForegroundColor Gray
    Write-Host "    .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Gray
    exit 1
}
Write-Ok "虚拟环境正常"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "hydrorag\server.py"))) {
    Write-Err "当前目录不像 HydroRAG 项目根目录：$ProjectRoot"
    exit 1
}

$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Warn2 "未找到 .env，将使用离线检索模式（不调用 DeepSeek）"
    Write-Step "如需生成回答，请复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY"
} else {
    $keyLine = Select-String -Path $envFile -Pattern "^\s*DEEPSEEK_API_KEY\s*=\s*\S+" -ErrorAction SilentlyContinue
    if ($keyLine) {
        Write-Ok "已配置 DeepSeek API Key"
    } else {
        Write-Warn2 ".env 中未填写 DEEPSEEK_API_KEY，回答将由检索片段直接拼出"
    }
}

# -------------------------------------------------- 2. 处理端口占用与复用 ---
Write-Host ""
Write-Host "[2/4] 检查端口 $Port" -ForegroundColor White

if (Test-ServiceReady $Port) {
    Write-Ok "检测到 HydroRAG 已在运行，直接复用"
    if (-not $NoBrowser) {
        Write-Step "打开 $BaseUrl"
        Start-Process $BaseUrl
    }
    Write-Host ""
    Write-Host "服务地址：$BaseUrl" -ForegroundColor Cyan
    exit 0
}

$occupied = Get-HydroRagProcesses $Port
if ($occupied) {
    Write-Warn2 "端口 $Port 被占用，正在释放"
    foreach ($proc in $occupied) {
        Write-Step "结束进程 $($proc.Id) ($($proc.ProcessName))"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 800
    if (Get-HydroRagProcesses $Port) {
        Write-Err "端口 $Port 仍被占用，请改用 -Port 指定其他端口。"
        exit 1
    }
}
Write-Ok "端口 $Port 可用"

# ------------------------------------------------------------ 3. 启动服务 ---
Write-Host ""
Write-Host "[3/4] 启动服务" -ForegroundColor White

$arguments = @("-m", "uvicorn", "hydrorag.server:app", "--host", "127.0.0.1", "--port", "$Port")
if ($Reload) { $arguments += "--reload" }

Write-Step "命令：python $($arguments -join ' ')"
$logDir = Join-Path $ProjectRoot "data\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "server.log"

# 在独立窗口启动，便于 Ctrl+C 停止；输出同时写入日志文件。
$command = "& '$VenvPython' $($arguments -join ' ') 2>&1 | Tee-Object -FilePath '$logFile'"
$serverProcess = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
    -WorkingDirectory $ProjectRoot -PassThru

Write-Step "服务进程已拉起（PID $($serverProcess.Id)）"

# 轮询等待就绪
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Milliseconds 500
    if (Test-ServiceReady $Port) { $ready = $true; break }
    if ($i % 10 -eq 0) { Write-Step "等待中就绪中…（$([int]($i * 0.5)) 秒）" }
}

if (-not $ready) {
    Write-Err "服务在 30 秒内未就绪。"
    Write-Host ""
    Write-Host "  请查看日志排查：$logFile" -ForegroundColor Yellow
    Write-Host "  常见原因：依赖未装全、端口冲突、Python 语法错误。" -ForegroundColor Yellow
    exit 1
}
Write-Ok "服务已就绪"

# ------------------------------------------------------ 4. 打开前端并汇报 ---
Write-Host ""
Write-Host "[4/4] 打开前端界面" -ForegroundColor White

try {
    $health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 5
    Write-Ok "资料数：$($health.documents) 份   分块数：$($health.chunks)"
    Write-Step "Embedding：$($health.embedding_backend) / $($health.embedding_model)"
    Write-Step "Reranker ：$($health.reranker_enabled) / $($health.reranker_model)"
    if ($health.deepseek_configured) {
        Write-Ok "DeepSeek：已连接（$($health.model)）"
    } else {
        Write-Warn2 "DeepSeek：未配置，当前为离线检索模式"
    }
} catch {
    Write-Warn2 "健康检查读取失败，但服务已响应。"
}

if (-not $NoBrowser) {
    Write-Step "打开 $BaseUrl"
    Start-Process $BaseUrl
}

Write-Host ""
Write-Host ("=" * 46) -ForegroundColor DarkGray
Write-Host " 前端界面 : $BaseUrl" -ForegroundColor Cyan
Write-Host " API 文档 : $BaseUrl/docs" -ForegroundColor Cyan
Write-Host " 健康检查 : $HealthUrl" -ForegroundColor Cyan
Write-Host ("=" * 46) -ForegroundColor DarkGray
Write-Host ""
Write-Host " 服务在新窗口中运行，按 Ctrl+C 可停止。" -ForegroundColor Gray
Write-Host " 或在本目录执行：.\start.ps1 -Stop" -ForegroundColor Gray
Write-Host ""
