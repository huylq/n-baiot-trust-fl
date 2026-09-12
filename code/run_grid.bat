@echo off
REM Run G1-G4 in separate processes, with one retry per group.
REM For indexed outputs use scripts/run_grid.py from the repository root.
cd /d "%~dp0"

for %%G in (G1 G2 G3 G4) do (
    echo === Nhom %%G - lan 1 ===
    python run_experiments.py --groups %%G
    if errorlevel 1 (
        echo === Nhom %%G that bai, thu lai lan 2 ===
        python run_experiments.py --groups %%G
    )
)
echo.
echo ==== HOAN TAT. Ket qua trong outputs\runs\ ====
pause
