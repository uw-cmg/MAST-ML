export PYTHONPATH=../../:$PYTHONPATH 

rm -rf output Ran*
python3 fit.py
mv Ran* output
python3 predict.py
