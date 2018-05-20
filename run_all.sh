#!/bin/bash
python3 data_split.py
python3 feature_extract.py
python3 gen_data.py
python3 dl.py
