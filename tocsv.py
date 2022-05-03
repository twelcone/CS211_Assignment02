import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = 'logs'
SAVE_DIR = 'csv'
folders = os.listdir(LOG_DIR)
for folder in folders:
    f = os.path.join(LOG_DIR, folder)
    event_acc = EventAccumulator(f)
    event_acc.Reload()
    # Show all tags in the log file
    # print(event_acc.Tags())

    _ , step, AvgRew = zip(*event_acc.Scalars('AvgRew'))
    _ , step, AvgEpLen = zip(*event_acc.Scalars('AvgEpLen'))

    df = pd.DataFrame(list(zip(step, AvgRew, AvgEpLen)), columns=['Step', 'AvgRew', 'AvgEpLen'])
    df = df.fillna(0)
    df.to_csv(SAVE_DIR + '/' + str(folder) + '.csv', index=False)
