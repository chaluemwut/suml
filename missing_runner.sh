#!/bin/sh
python missing_attribute.py bagging &
python missing_attribute.py boosted &
python missing_attribute.py randomforest &
python missing_attribute.py nb &
python missing_attribute.py decsiontree &