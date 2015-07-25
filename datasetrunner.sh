#!/bin/sh
python parallel_manual_by_dataset.py bagging &
python parallel_manual_by_dataset.py boosted &
python parallel_manual_by_dataset.py randomforest &
python parallel_manual_by_dataset.py nb &
python parallel_manual_by_dataset.py decsiontree &