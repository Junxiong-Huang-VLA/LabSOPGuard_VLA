# Environment Management

Policy:

- Exactly one virtual environment for this project.
- Environment name must equal project name: `LabEmbodiedVLA`.
- If env exists, reuse it.
- Do not create additional environment names.

Commands:

```bash
# Create or reuse
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -ProjectName LabEmbodiedVLA -PythonVersion 3.10 -InstallMethod both

# Activate
conda activate LabEmbodiedVLA

# Check
python 14_check_environment.py --project-name LabEmbodiedVLA

# Remove
conda env remove -n LabEmbodiedVLA

# Rebuild
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -ProjectName LabEmbodiedVLA -PythonVersion 3.10 -InstallMethod both
```
