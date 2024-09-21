#!/bin/sh

echo "Training and Test errors for car dataset"
python3 car.py

echo "Results for bank dataset -unknown as feature value"
python3 bank.py

echo "Results for bank dataset -unknown as missing"
python3 bank_missing.py