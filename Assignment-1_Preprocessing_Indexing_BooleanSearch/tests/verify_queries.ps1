$dir = Split-Path -Parent $PSScriptRoot
$binary = "$dir\bin\MaamTeam_pa1.exe"
$index = "$dir\MaamTeam_cran.index"
$stops = "$dir\data\stopwords.txt"
$result = "$dir\MaamTeam_query.result"

Push-Location $dir

$queries = @(
    @{n=1; t1="aeroelastic"; t2="aircraft"; a=5; o=84},
    @{n=2; t1="dynamics"; t2="effects"; a=34; o=586},
    @{n=3; t1="hypersonic"; t2="wake"; a=5; o=211},
    @{n=4; t1="flutter"; t2="steady"; a=11; o=151},
    @{n=5; t1="viscosity"; t2="reynolds"; a=23; o=239},
    @{n=6; t1="heat"; t2="stagnation"; a=80; o=360},
    @{n=7; t1="oscillatory"; t2="transonic"; a=1; o=83},
    @{n=8; t1="creep"; t2="buckling"; a=26; o=136},
    @{n=9; t1="pressure"; t2="wing"; a=91; o=687},
    @{n=10; t1="transonic"; t2="nozzle"; a=3; o=140},
    @{n=11; t1="excitation"; t2="noise"; a=5; o=47},
    @{n=12; t1="mass"; t2="flutter"; a=6; o=121}
)

if (-not (Test-Path $binary)) { Write-Host "ERROR: Binary not found"; exit 1 }
if (-not (Test-Path $index)) { Write-Host "ERROR: Index not found"; exit 1 }

Write-Host ""
Write-Host "Verifying $($queries.Count) queries:"
Write-Host ""

$p = 0
$f = 0

for ($i = 0; $i -lt $queries.Count; $i++)
{
    $q = $queries[$i]
    Write-Host "Query $($q.n): '$($q.t1) AND $($q.t2)'" 
    
    & $binary search $index $stops "$($q.t1) AND $($q.t2)" "MaamTeam" | Out-Null
    $ac = ((Get-Content $result).Trim() -split " ").Count
    $a_ok = ($ac -eq $q.a)
    
    & $binary search $index $stops "$($q.t1) OR $($q.t2)" "MaamTeam" | Out-Null
    $oc = ((Get-Content $result).Trim() -split " ").Count
    $o_ok = ([Math]::Abs($oc - $q.o) -le 1)
    
    if ($a_ok)
    {
        Write-Host "  AND: $ac (expected $($q.a)) [PASS]" -ForegroundColor Green -NoNewline
    }
    else
    {
        Write-Host "  AND: $ac (expected $($q.a)) [FAIL]" -ForegroundColor Red -NoNewline
        $f = $f + 1
    }
    
    Write-Host " | " -NoNewline
    
    if ($o_ok)
    {
        Write-Host "OR: $oc (expected $($q.o)) [PASS]" -ForegroundColor Green
        $p = $p + 2
    }
    else
    {
        Write-Host "OR: $oc (expected $($q.o)) [FAIL]" -ForegroundColor Red
        $f = $f + 1
    }
}

Write-Host ""
Write-Host "Result: $p/$($queries.Count * 2) passed"
if ($f -eq 0) { Write-Host "PASS" -ForegroundColor Green } else { Write-Host "FAIL ($f errors)" -ForegroundColor Red }

Pop-Location
