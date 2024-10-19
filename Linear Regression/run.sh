#!/bin/sh

echo "Result of batch_gradienet_descent"
python3 batch_descent.py

echo "Results for stochastic_gradient_descent"
python3 gradient_descent.py

echo "Results for optimalweight"
python3 optimal.py