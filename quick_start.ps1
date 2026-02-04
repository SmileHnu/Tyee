# Quick start script for Tyee (Windows PowerShell)
# This script prepares data and runs the MIT-BIH experiment.
# Comments are in English; prompts are in Chinese for clarity.

Param(
    [string]$ConfigRelative = "tyee/config/mit_bih.yaml"
)

# Get script and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path $ScriptDir

Write-Host "=== Tyee Quick Start: MIT-BIH Arrhythmia Detection Experiment ===`n"

# Step 1: Create data directory
$DestDir = Join-Path $ProjectRoot "data" "original"
Write-Host "Step 1/3: Creating data directory..."
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir | Out-Null
}
Write-Host "Data directory created: $DestDir"

# Step 2: Download dataset
$DataUrl = "https://physionet.org/files/mitdb/1.0.0/"
Write-Host "Step 2/3: Checking and downloading MIT-BIH dataset..."

# Simple check if dest already has files
if ((Get-ChildItem -Path $DestDir -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0) {
    Write-Host "检测到已有数据，跳过下载步骤： $DestDir"
} else {
    Write-Host "Downloading dataset to $DestDir ..."

    try {
        # Try Invoke-WebRequest (PowerShell)
        Invoke-WebRequest -Uri $DataUrl -OutFile (Join-Path $DestDir ([System.IO.Path]::GetFileName($DataUrl))) -UseBasicParsing
        Write-Host "Dataset download initiated (single file). Note: for recursive mirror, consider using WSL or wget on Windows."
    } catch {
        Write-Host "下载失败：$($_.Exception.Message)"
        Write-Host "建议在 Windows 上使用 WSL 或在 Linux/macOS 上执行该脚本以完整镜像数据。"
    }
}

# Step 3: Run training
Write-Host "Step 3/3: Starting model training..."
Push-Location $ProjectRoot
$python = "python"
if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    Write-Host "错误：找不到 Python 命令。请确保已安装 Python 并在 PATH 中，或激活 Conda 环境。"
    Pop-Location
    exit 1
}

$ConfigPath = Join-Path $ProjectRoot $ConfigRelative
Write-Host "运行命令: $python train.py -c $ConfigPath"
& $python "train.py" "-c" $ConfigPath

Pop-Location

Write-Host "=== Experiment completed! ==="
Write-Host "Results saved in $ProjectRoot\experiments\ directory"
Write-Host "Use 'tensorboard --logdir $ProjectRoot\experiments\ --port 6006' to view training progress"