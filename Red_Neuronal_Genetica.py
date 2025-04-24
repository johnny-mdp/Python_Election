import numpy as np
import Datos as Datos


import torch
import torch.nn as nn

import torch.optim as optim

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.model(x)


def save_model():
	filepath = input("Enter the filepath where the model will be saved.\n")
	model = Net()
	torch.save(model.state_dict(), filepath)

# Load model weights
def load_model():
	filepath = input("Enter the filepath to the model.\n")
	model = Net()
	model.load_state_dict(torch.load(filepath))
	model.eval()  # Set to evaluation mode
