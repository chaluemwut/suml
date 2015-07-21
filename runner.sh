#!/bin/sh
python parallel_manual.py bagging &
python parallel_manual.py boosted &
python parallel_manual.py randomforest &
python parallel_manual.py nb &
python parallel_manual.py knn &
python parallel_manual.py decsiontree &