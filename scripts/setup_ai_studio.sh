#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U -r requirements.txt
modelscope download --model google/gemma-3-270m-it --local_dir ./gemma3-270m-it
python student_kit/audit_reward.py --output reward_calibration.json
python -m pytest -q

echo "Setup complete. Model: ./gemma3-270m-it"
