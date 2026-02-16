#Requires -Version 5.1
<#
.SYNOPSIS
    Run a full stratum-lab behavioral scan pipeline.
.DESCRIPTION
    PowerShell equivalent of run_scan.sh. Performs preflight checks (vLLM, Docker,
    structural scans, runner image) then invokes the stratum-lab pipeline command.
.PARAMETER StructuralScanDir
    Path to the directory containing structural scan JSON files.
.PARAMETER OutputDir
    Path to the output directory. Defaults to ./data/scan-output.
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$StructuralScanDir,

    [Parameter(Position = 1)]
    [string]$OutputDir = "./data/scan-output"
)

$ErrorActionPreference = "Stop"

# ===================== CONFIGURATION =====================
$VllmUrl      = if ($env:VLLM_URL)              { $env:VLLM_URL }              else { "http://localhost:8000/v1" }
$VllmModel    = if ($env:STRATUM_VLLM_MODEL)    { $env:STRATUM_VLLM_MODEL }    else { "Qwen/Qwen2.5-72B-Instruct" }
$Target       = if ($env:SCAN_TARGET)            { $env:SCAN_TARGET }            else { "1000" }
$Concurrent   = if ($env:SCAN_CONCURRENT)        { $env:SCAN_CONCURRENT }        else { "8" }
$Timeout      = if ($env:SCAN_TIMEOUT)           { $env:SCAN_TIMEOUT }           else { "600" }
$PilotSize    = if ($env:SCAN_PILOT_SIZE)        { $env:SCAN_PILOT_SIZE }        else { "20" }

$env:STRATUM_VLLM_MODEL = $VllmModel

# ===================== PREFLIGHT =====================
Write-Host "=== Preflight ==="

# -- vLLM endpoint --
Write-Host -NoNewline "  vLLM endpoint ($VllmUrl)... "
try {
    $null = Invoke-WebRequest -Uri "$VllmUrl/models" -UseBasicParsing -ErrorAction Stop
    Write-Host "OK"
} catch {
    Write-Host "FAIL"
    exit 1
}

# -- Docker --
Write-Host -NoNewline "  Docker... "
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "docker info failed" }
    Write-Host "OK"
} catch {
    Write-Host "FAIL"
    exit 1
}

# -- Structural scans --
Write-Host -NoNewline "  Structural scans... "
$scanFiles = Get-ChildItem -Path $StructuralScanDir -Filter "*.json" -Recurse -File
$scanCount = $scanFiles.Count
Write-Host "$scanCount files"
if ($scanCount -eq 0) {
    Write-Host "ERROR: no scan files"
    exit 1
}

# -- Runner image --
Write-Host -NoNewline "  Runner image... "
$null = docker image inspect stratum-lab-runner 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK"
} else {
    Write-Host "building..."
    stratum-lab build-image
}

Write-Host ""

# ===================== SCAN =====================
Write-Host "=== Starting scan: target=$Target, concurrent=$Concurrent, timeout=${Timeout}s ==="

# Ensure the output directory exists so the log file can be written
if (-not (Test-Path $OutputDir)) {
    $null = New-Item -ItemType Directory -Path $OutputDir -Force
}

$logFile = Join-Path $OutputDir "scan.log"

stratum-lab pipeline `
    --input-dir $StructuralScanDir `
    --output-dir $OutputDir `
    --target $Target `
    --vllm-url $VllmUrl `
    --concurrent $Concurrent `
    --timeout $Timeout `
    --pilot `
    --pilot-size $PilotSize `
    --max-instrumentation-failure-rate 0.20 `
    --max-model-failure-rate 0.15 `
    --resume `
    2>&1 | Tee-Object -FilePath $logFile

# ===================== VALIDATE =====================
Write-Host ""
Write-Host "=== Post-scan validation ==="
python scripts/validate_scan.py $OutputDir
