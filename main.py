from test import test
from train import train
from model import Model
import torch
from plot import plot_training_loss, plot_validation_loss, plot_success_rate, plot_validation_success_rate

model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)

model, loss_history, success_history, val_loss_history, val_success_history = train(model, 10)
plot_training_loss(loss_history)
plot_validation_loss(val_loss_history)
plot_success_rate(success_history)  
plot_validation_success_rate(val_success_history)
