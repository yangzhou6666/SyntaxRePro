
mkdir temp
mkdir logs
mkdir data/results

echo 'make direction finished'

echo 'Make sure you have (prutor-deepfix-09-12-2017.zip) in data/'
# wget https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip -P data/
cd data
unzip prutor-deepfix-09-12-2017.zip
mv prutor-deepfix-09-12-2017/* iitk-dataset/
rm -rf prutor-deepfix-09-12-2017
cd iitk-dataset/
gunzip prutor-deepfix-09-12-2017.db.gz
mv prutor-deepfix-09-12-2017.db dataset.db
cd ../..

echo 'Preprocessing DeepFix dataset...'
export PYTHONPATH=.
python data_processing/preprocess.py

echo "Make sure that your tensorflow is version 1.0.1 before proceeding, checking your tensorflow version now!"
echo
python -c 'import tensorflow as tf; print tf.__version__'

echo 'init.sh executed finished'