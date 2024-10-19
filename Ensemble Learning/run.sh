#!/bin/sh

echo "Results for Adaboost"
python3 adaboost.py

echo "Results for bagged tree"
python3 bagged.py

echo "Results for biasvariance"
python3 bias_var.py

echo "Results for randomforest"
python3 test.py

echo "Results for bias variance for Random Tree vs Bagged Tree vs Single Tree"
python3 bias_var_2.py