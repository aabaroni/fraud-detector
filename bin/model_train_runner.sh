today=`date +%Y-%m-%d.%H:%M:%S`

cd ../fraud-detector/
echo "In progress.. Logs will be written out to ../logs/log_${today}.txt"
mkdir -p ../logs/ # Create logs folder if not exist
python model_train_comparison.py $1 > ../../logs/log_outdet_${today}.txt