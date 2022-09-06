import csv

filename = open('output/test/history_log.csv', 'r')
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
# plot loss
ax = plt.figure(figsize=(4, 3)).add_subplot(111)
ax.plot(epoch, loss, color='orange', linestyle='-', linewidth=1)
ax.plot(epoch, val_loss, color='blue', linestyle='-', linewidth=1)
ax.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim([0, 1.1])
plt.xlim([-1, 100])
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# plot dice coef
ax = plt.figure(figsize=(4, 3)).add_subplot(111)
ax.plot(epoch, dice_coef, color='orange', linestyle='-', linewidth=1)
ax.plot(epoch, val_dice_coef, color='blue', linestyle='-', linewidth=1)
ax.grid()
plt.ylabel('dice coefficient')
plt.xlabel('epoch')
plt.ylim([0, 1.1])
plt.xlim([-1, 100])
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

