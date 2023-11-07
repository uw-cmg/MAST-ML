# Start from clean environment to prevent package conflicts
python3 -m venv python_env
source python_env/bin/activate
pip install -U pip

# Install mastml (in this case a specific branch)
pip install git+https://github.com/uw-cmg/MAST-ML.git@dev_lane

# Remove junk if this script was ran in the past
rm -rf domain_run calibration_run Ran*
python3 fit.py  # Run a cript to provide a fit model, calibration, and domain
