import subprocess
import time
import os

NUMGAMES = 150

bash = 'C:\\Windows\\System32\\bash.exe'
work_dir = "E:\\PyCharmProjects\\RL\\to repo\\"
host = 'build.sh'
agent = 'my_player3.py'
start_time = time.time()
first_epoch_start = 0
print("Engine script starts.")

for epoch in range(NUMGAMES):
    print("Epoch: {}".format(epoch))
    try:
        os.remove('winner_file.txt')
    except FileNotFoundError:
        print("No winner file.")
    if epoch == 0:
        first_epoch_start = time.time()

    subprocess.call([bash, host, agent], cwd=work_dir)

    if epoch == 0:
        print("ETA for {} games: {} h.".format(NUMGAMES, (NUMGAMES * (time.time() - first_epoch_start)) / 3600))

print("Executed: {} minutes".format((time.time() - start_time) / 60))

with open('move_time', 'r') as mt:
    line = mt.readline()
    potential_losses = 0
    sum = 0
    line_num = 0
    while line:
        sec = float(line)
        line_num += 1
        if sec > 8:
            print("WILL PROBABLY BE INTERUPTED HERE")
            potential_losses += 1
        sum += sec
        line = mt.readline()
    print("Avg. move time: {}".format(sum / line_num))
os.remove('move_time')
