#!/bin/bash
#SBATCH -A naiss2026-4-1165-gpu
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -t 12:00:00

set -e

cd "/nobackup/proj/disk/naiss2026-4-1165/personal/wanner/seeqst_shadows/"
source .venv/bin/activate

python -u run_experiment.py --config-dir=cfgs_arrhenius --config-name=arrhenius_xy_local_paulis_manyobs.yaml