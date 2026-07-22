# Continuous Live-Sync of thermodynamic telemetry from the remote Vast AI substrate.
# Uses rsync (available in modern Windows/WSL) for efficient delta-transfers.

$RemoteUser = "root"
$RemoteHost = "ssh1.vast.ai"
$RemotePort = "26117"
$RemotePath = "/root/HENRI/telemetry_logs/"
$LocalPath = ".\HENRI V2\telemetry_logs\"

Write-Host "[ALETHEIA] Initiating Continuous Thermodynamic Telemetry Stream..." -ForegroundColor Cyan

if (-not (Test-Path -Path $LocalPath)) {
    New-Item -ItemType Directory -Path $LocalPath | Out-Null
}

# Ensure ssh pathing works natively with rsync
$sshCommand = "ssh -p $RemotePort -o StrictHostKeyChecking=no"

while ($true) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Syncing active wavefronts..." -ForegroundColor DarkGray
    
    # Execute delta-sync (pulls only new lines appended to JSONL files)
    # Using WSL rsync if native is unavailable; fallback to scp if necessary.
    try {
        rsync -avz -e $sshCommand --update "$RemoteUser@${RemoteHost}:$RemotePath" "$LocalPath" | Out-Null
    } catch {
        Write-Warning "rsync failed, falling back to SCP loop..."
        scp -o StrictHostKeyChecking=no -P $RemotePort -r "$RemoteUser`@${RemoteHost}:$RemotePath*" $LocalPath | Out-Null
    }

    Start-Sleep -Seconds 5
}