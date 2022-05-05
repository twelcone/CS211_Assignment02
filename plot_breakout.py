import csv
import pandas as pd
import matplotlib.pyplot as plt

Breakout_DQN = pd.read_csv('csv/Breakout-v0_DQN.csv')
Breakout_DoubleDQN = pd.read_csv('csv/Breakout-v0_DoubleDQN.csv')
Breakout_DuelingDQN = pd.read_csv('csv/Breakout-v0_DuelingDQN.csv')

DQN_STEP = Breakout_DQN['Step'].tolist()
DOUBLE_DQN_STEP = Breakout_DoubleDQN['Step'].tolist()
DUELING_DQN_STEP = Breakout_DuelingDQN['Step'].tolist()

Breakout_DoubleDQN = Breakout_DoubleDQN[:-(len(DOUBLE_DQN_STEP) - len(DQN_STEP))]
DOUBLE_DQN_STEP = Breakout_DoubleDQN['Step'].tolist()
Breakout_DuelingDQN = Breakout_DuelingDQN[:-(len(DUELING_DQN_STEP) - len(DQN_STEP))]
DUELING_DQN_STEP = Breakout_DuelingDQN['Step'].tolist()

DQN_AVGREW = Breakout_DQN["AvgRew"].tolist()
DOUBLE_DQN_AVGREW = Breakout_DoubleDQN['AvgRew'].tolist()
DUELING_DQN_AVGREW = Breakout_DuelingDQN['AvgRew'].tolist()

DQN_AVGEPLEN = Breakout_DQN["AvgEpLen"].tolist()
DOUBLE_DQN_AVGEPLEN = Breakout_DoubleDQN['AvgEpLen'].tolist()
DUELING_DQN_AVGEPLEN = Breakout_DuelingDQN['AvgEpLen'].tolist()

def plot_avgrew():
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.plot(DQN_STEP, DQN_AVGREW, label="DQN")
    plt.plot(DQN_STEP, DOUBLE_DQN_AVGREW, label="DoubleDQN")
    plt.plot(DQN_STEP, DUELING_DQN_AVGREW, label="DuelingDQN")

    plt.grid()
    plt.legend(loc="lower right")
    plt.title('Breakout-v0 AvgRew Comparison')
    plt.show()

def plot_avgEpLen():
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.plot(DQN_STEP, DQN_AVGEPLEN, label="DQN")
    plt.plot(DQN_STEP, DOUBLE_DQN_AVGEPLEN, label="DoubleDQN")
    plt.plot(DQN_STEP, DUELING_DQN_AVGEPLEN, label="DuelingDQN")

    plt.grid()
    plt.legend(loc="lower right")
    plt.title('Breakout-v0 AvgEpLen Comparison')
    plt.show()

plot_avgEpLen()