#!/usr/bin/env bash
# One-command reproduction of the DIVINO pipeline (reduced-scale defaults).
set -e
PY=${PY:-python3}
echo "[1/7] validate physics vs body-fitted FEM";  $PY validate_physics.py
echo "[2/7] train operator (PINO)";                $PY train.py --N 128 --steps 9000 --out checkpoints/pino.pt
echo "[3/7] forward benchmark (T1,F1)";            $PY benchmark_forward.py --n_indist 150 --n_ood 60
echo "[4/7] baselines & speedup (T2,T3)";          $PY baselines.py --ntest 40 --steps 3000
echo "[5/7] inverse study (T4,F2,F3)";             $PY inverse_study.py --reps 20
echo "[6/7] insulation fidelity (F5)";             $PY insulation_fidelity.py
echo "[7/7] field comparison figure (F7)";         $PY make_figures.py
echo "ablation (T5,F4) [optional, slow]";          $PY ablation.py --steps 2500 || true
echo "DONE -> results/*.csv, figures/*.png"
