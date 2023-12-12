import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV file
input_csv_path = './output/20231211_2d_unet_maximilian_adam_lr0001_filter16_batchsize10_epoch100_earlystop12_augmentationFalse/history_log.csv' #'path/to/your/file.csv'
data = pd.read_csv(input_csv_path, sep='\s+')
output_plot_path= os.path.join(os.path.dirname(input_csv_path), "plot_dice_loss.png")

# Extract relevant columns
epochs = data['epoch']
train_dice_coef = data['dice_coef']
train_loss = data['loss']
val_dice_coef = data['val_dice_coef']
val_loss = data['val_loss']

# Plot graphs
plt.figure(figsize=(12, 6))

# Plot for Dice Coefficient
plt.subplot(1, 2, 1)
plt.plot(epochs, train_dice_coef, label='Train Dice Coef')
plt.plot(epochs, val_dice_coef, label='Validation Dice Coef')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()

# Plot for Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.tight_layout()
plt.savefig(output_plot_path)
plt.show()
