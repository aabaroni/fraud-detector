###################################
# Accumulator job scans for any new transactions in the
# target folder and processes at a specified interval
###################################

import os
from datetime import datetime
from api_scorer import score_api
import time

# scan frequency
scan_freq_seconds = 1

# initial file scan
files_earlier = []
files_now = os.listdir("../data/automation_in")

# Infinite loop, will continue until user types ctrl-C to quit the application
while True:
    if len(files_now) > len(files_earlier):
        print("New transactions detected. Time:", datetime.now())
        # list files to process
        to_process = [x for x in files_now if x not in files_earlier]
        print("Processing files: ", to_process)

        # update list
        files_earlier = files_now

        # call api with new file
        score_api(to_process)

    files_now = os.listdir("../data/automation_in")
    time.sleep(scan_freq_seconds)