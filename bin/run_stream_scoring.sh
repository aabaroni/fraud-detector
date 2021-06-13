today=`date +%Y-%m-%d.%H.%M.%S`

echo "Starting up fastapi application in the background. Give it a few seconds.."
# kill all processes on port 8000 to start fresh:
lsof -ti:8000 | xargs kill -9
cd ../api/
uvicorn app:app --reload &
sleep 10

echo "Starting up accumulator - will scan through transactions and score any new ones."
echo "Use ctrl + c to quit at any point"
echo "In progress.. Logs will be written out to ../logs/automation_log_${today}.txt"
cd ../automation/
mkdir -p ../logs/ # Create logs folder if not exist
python accumulator.py $1 > ../logs/automation_log_${today}.txt
cd ../bin

