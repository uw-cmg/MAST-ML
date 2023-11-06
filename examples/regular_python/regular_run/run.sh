export PYTHONPATH=$(pwd)/../../../:$PYTHONPATH 

rm -rf domain_run calibration_run Ran*
python3 domain_fit.py
