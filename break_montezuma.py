import csv
import pandas as pd
import matplotlib.pyplot as plt

Montezuma_DQN = pd.read_csv('csv/MontezumaRevenge-v0_DQN.csv')
Montezuma_DoubleDQN = pd.read_csv('csv/MontezumaRevenge-v0_DoubleDQN.csv')
Montezuma_DuelingDQN = pd.read_csv('csv/MontezumaRevenge-v0_DuelingDQN.csv')

DQN_STEP = Montezuma_DQN['Step'].tolist()
DOUBLE_DQN_STEP = Montezuma_DoubleDQN['Step'].tolist()
DUELING_DQN_STEP = Montezuma_DuelingDQN['Step'].tolist()

Montezuma_DoubleDQN = Montezuma_DoubleDQN[:-(len(DOUBLE_DQN_STEP) - len(DUELING_DQN_STEP))]
DOUBLE_DQN_STEP = Montezuma_DoubleDQN['Step'].tolist()
Montezuma_DQN = Montezuma_DQN[:-(len(DQN_STEP) - len(DUELING_DQN_STEP))]
DQN_STEP = Montezuma_DQN['Step'].tolist()

DQN_AVGREW = Montezuma_DQN["AvgRew"].tolist()
DOUBLE_DQN_AVGREW = Montezuma_DoubleDQN['AvgRew'].tolist()
DUELING_DQN_AVGREW = Montezuma_DuelingDQN['AvgRew'].tolist()

DQN_AVGEPLEN = Montezuma_DQN["AvgEpLen"].tolist()
DOUBLE_DQN_AVGEPLEN = Montezuma_DoubleDQN['AvgEpLen'].tolist()
DUELING_DQN_AVGEPLEN = Montezuma_DuelingDQN['AvgEpLen'].tolist()

def plot_avgrew():
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.plot(DQN_STEP, DQN_AVGREW, label="DQN")
    plt.plot(DQN_STEP, DOUBLE_DQN_AVGREW, label="DoubleDQN")
    plt.plot(DQN_STEP, DUELING_DQN_AVGREW, label="DuelingDQN")

    plt.grid()
    plt.legend(loc="lower right")
    plt.title('MontezumaRevenge-v0 AvgRew Comparison')
    plt.show()

def plot_avgEpLen():
    plt.rcParams["figure.figsize"] = (15, 6)
    plt.plot(DQN_STEP, DQN_AVGEPLEN, label="DQN")
    plt.plot(DQN_STEP, DOUBLE_DQN_AVGEPLEN, label="DoubleDQN")
    plt.plot(DQN_STEP, DUELING_DQN_AVGEPLEN, label="DuelingDQN")

    plt.grid()
    plt.legend(loc="lower right")
    plt.title('MontezumaRevenge-v0 AvgEpLen Comparison')
    plt.show()

plot_avgEpLen()