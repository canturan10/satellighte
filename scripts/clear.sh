find . | grep -E '(__pycache__|\.pyc|\.pyo$|checkpoints|lightning_logs)' | xargs rm -rf
