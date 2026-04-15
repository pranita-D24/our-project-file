$workingDir = "c:\Trivim Internship\engineering_comparison_system"
$drawingDir = "..\drawing files"

Write-Host "=== STARTING POWERSHELL-DIRECT IMAGE AUDIT ==="

# 1. Gather v1 files
$v1Files = Get-ChildItem -Path $drawingDir -Filter "*_V1.png"

foreach ($v1 in $v1Files) {
    $v2Name = $v1.Name -replace "_V1", "_V2"
    $v2Path = Join-Path $drawingDir $v2Name
    
    if (Test-Path $v2Path) {
        $id = $v1.Name -replace "_V1.png", ""
        Write-Host "Comparing $id..."
        
        # 2. Call Python Driver
        & "$workingDir\venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'$workingDir'); from img_audit import compare, ReportGenerator, cv2, np; res = compare(r'$($v1.FullName)', r'$v2Path'); print(f'Similarity: {res.similarity}')"
    } else {
        Write-Host "Skipping $($v1.Name): V2 not found."
    }
}
