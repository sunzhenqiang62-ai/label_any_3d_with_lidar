# Copy LabelAny3D to remote workspace via scp (requires OpenSSH + key/password).
param(
    [string]$RemoteUser = "chengyue.sun",
    [string]$RemoteHost = "192.168.48.24",
    [string]$RemoteDir = "/mnt/cfs-baidu/public/chengyue.sun/workspace",
    [string]$ProjectName = "LabelAny3D"
)

$Src = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dest = "${RemoteUser}@${RemoteHost}:${RemoteDir}/${ProjectName}/"

Write-Host "Source:      $Src"
Write-Host "Destination: $Dest"

ssh "${RemoteUser}@${RemoteHost}" "mkdir -p '${RemoteDir}/${ProjectName}'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

scp -r `
  "$Src\README.md" `
  "$Src\docs" `
  "$Src\scripts" `
  "$Src\src" `
  "$Src\requirements.txt" `
  "$Src\requirements-py123d.txt" `
  "${RemoteUser}@${RemoteHost}:${RemoteDir}/${ProjectName}/"

Write-Host "Done."
