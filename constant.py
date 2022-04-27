GAME_NAME = 'MontezumaRevenge-v0'
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(1e6)
MIN_REPLAY_SIZE=50000
EPSILON_START=1.0
EPSILON_END=0.1
EPSILON_DECAY=int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000//NUM_ENVS
LR = 5e-5
SAVE_PATH = './model/ATARI_MODEL.pack'
SAVE_INTERVAL = 10000
LOG_DIR = './logs/atari_vannila'
LOG_INTERVAL = 1000