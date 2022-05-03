import csv
import pandas as pd
import matplotlib.pyplot as plt

Breakout_DQN = pd.read_csv('csv/Breakout-v0_DQN.csv')
Breakout_DoubleDQN = pd.read_csv('csv/Breakout-v0_DoubleDQN.csv')

DQN_STEP = Breakout_DQN['Step'].tolist()
DOUBLE_DQN_STEP = Breakout_DoubleDQN['Step'].tolist()

Breakout_DoubleDQN = Breakout_DoubleDQN[:-(len(DOUBLE_DQN_STEP) - len(DQN_STEP))]
DOUBLE_DQN_STEP = Breakout_DoubleDQN['Step'].tolist()

DQN_AVGREW = Breakout_DQN["AvgRew"].tolist()
DOUBLE_DQN_AVGREW = Breakout_DoubleDQN['AvgRew'].tolist()

plt.plot(DQN_STEP, DQN_AVGREW, label="DQN")
plt.plot(DQN_STEP, DOUBLE_DQN_AVGREW, label="DoubleDQN")
plt.grid()
plt.legend()
plt.show()
