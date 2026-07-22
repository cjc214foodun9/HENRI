# Syncs telemetry logs from the remote Vast AI server to the local workspace
$RemoteUser = "root"
$RemoteHost = "ssh1.vast.ai"
$RemotePort = "26117"
$RemotePath = "/root/HENRI/telemetry_logs/*"
$LocalPath = ".\HENRI V2\telemetry_logs"

Write-Host "[ALETHEIA] Synchronizing thermodynamic telemetry from Vast AI..." -ForegroundColor Cyan

# Create the local directory if it doesn't exist
if (-not (Test-Path -Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath | Out-Null
}

# Execute Secure Copy Protocol (SCP)
scp -o StrictHostKeyChecking=no -P $RemotePort -r "$RemoteUser`@${RemoteHost}:$RemotePath" $LocalPath

Write-Host "[ALETHEIA] Synchronization Complete." -ForegroundColor Green
