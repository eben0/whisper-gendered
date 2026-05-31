#Requires -Version 7.0
$ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
       Where-Object { $_.IPAddress -notlike '169.*' -and $_.IPAddress -ne '127.0.0.1' -and
                      $_.InterfaceAlias -notlike '*WSL*' -and $_.InterfaceAlias -notlike '*vEthernet*' -and
                      $_.InterfaceAlias -notlike '*Loopback*' } |
       Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "Server running at: http://${ip}:9000"
Write-Host ""
$env:PYTHONPATH = "$PSScriptRoot\src"
python src/server.py
