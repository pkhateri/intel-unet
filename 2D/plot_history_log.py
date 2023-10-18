import csv
import sys
import numpy as np
import os

if len(sys.argv)<2:
    print("Error: missing arument: path to the history log file")
filename = open(sys.argv[1], 'r')
foldername = os.path.dirname(sys.argv[1])
file = csv.DictReader(filename, delimiter=' ')

epoch=[]
dice_coef=[]
loss=[]
val_dice_coef=[]
val_loss=[]


for row in file:
    epoch.append(int(row['epoch']))
    dice_coef.append(float(row['dice_coef']))
    loss.append(float(row['loss']))
    val_dice_coef.append(float(row['val_dice_coef']))
    val_loss.append(float(row['val_loss']))

###########
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

#plt.style.use('classic')
# plot loss
ax = plt.figure(figsize=(4, 3)).add_subplot(111)
ax.plot(epoch, loss, color='orange', linestyle='-', linewidth=1)
ax.plot(epoch, val_loss, color='blue', linestyle='-', linewidth=1)
ax.grid()
#ax.tick_params(axis="both", which="both")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim([-1, 100])
plt.ylim([0, 1.1])
x_major_ticks = np.arange(0, 101, 10)
x_minor_ticks = np.arange(0, 101, 5)
y_major_ticks = np.arange(0, 1.1, 0.1)
y_minor_ticks = np.arange(0, 1.1, 0.02)
ax.set_xticks(x_major_ticks)
ax.set_xticks(x_minor_ticks, minor=True)
ax.set_yticks(y_major_ticks)
ax.set_yticks(y_minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.7)

plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(os.path.join(foldername,"loss.png"), bbox_inches='tight', dpi=1200)
plt.show()

# plot dice coef
ax = plt.figure(figsize=(4, 3)).add_subplot(111)
ax.plot(epoch, dice_coef, color='orange', linestyle='-', linewidth=1)
ax.plot(epoch, val_dice_coef, color='blue', linestyle='-', linewidth=1)
ax.grid()
plt.xlabel('epoch')
plt.ylabel('dice coefficient')
plt.xlim([-1, 100])
plt.ylim([0, 1.1])
x_major_ticks = np.arange(0, 101, 10)
x_minor_ticks = np.arange(0, 101, 5)
y_major_ticks = np.arange(0, 1.1, 0.1)
y_minor_ticks = np.arange(0, 1.1, 0.02)
ax.set_xticks(x_major_ticks)
ax.set_xticks(x_minor_ticks, minor=True)
ax.set_yticks(y_major_ticks)
ax.set_yticks(y_minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.7)

plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig(os.path.join(foldername,"dice_coefficient.png"), bbox_inches='tight', dpi=1200)
plt.show()

