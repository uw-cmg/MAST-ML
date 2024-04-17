#!/bin/bash

# Python packages
export PYTHONPATH=../../../../:$PYTHONPATH
export PYTHONPATH=../../../../../materials_application_domain_machine_learning/src/:$PYTHONPATH
export PYTHONPATH=../../../../../transfernet/src:$PYTHONPATH

# Python script
python3 fit.py
