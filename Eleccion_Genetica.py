import numpy as np
import Red_Neuronal_Genetica as RNG
import Datos as Datos
	
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

def save_model(model, filepath):
	torch.save(model.state_dict(), filepath)

# Load model weights
def load_model(model, filepath):
	model.load_state_dict(torch.load(filepath))
	model.eval()  # Set to evaluation mode


def predict_new_election(model):
	c = 's'
	filepath = "Election_predictions.txt"
	with open(filepath, "w") as f:
		while c=='s':
			elect, argentina, candidates = Datos.ReadData()
	
			predictions = []
			p = 0
			for i in range(len(elect)):
				tensor_input = torch.tensor(elect[i], dtype=torch.float32).unsqueeze(0)  # Add batch dim
				pred = model(tensor_input).item()
				predictions.append(pred)
				p = p + pred
			
			print(p)
			for i in range( len(predictions)):
				predictions[i] /= p
				print(f"Candidate {i+1}: predicted vote share = {predictions[i]:.4f}")
		
			candidates["Results"] = predictions
			l = input("Save Predictions? y/n\n")
			# Si se tiene un archivo que se quiere usar no tiene sentido volver a entrenar otro.
			while ( l.lower() != 'y' and l.lower() != 'n'):
				l = input("Incorrect, try again.\n")
		
			if ( l.lower() == 'y'):
				filepath = "Election_predictions.txt"
				Datos.SaveElection(argentina,candidates,f)
			c = input("Simulate another election? y/n\n")
			while ( c.lower() != 'y' and c.lower() != 'n'):
				c = input("Incorrect, try again.\n")
			
	if os.path.isfile(filepath) and os.stat(filepath).st_size == 0:
		os.remove(filepath)
		print(f"{filepath} was empty and has been deleted.")
	else:
		print(f"{filepath} is not empty or does not exist.")
	return




loaded = input("Do you have an already trained NN you wish to load? y/n\n")
# Si se tiene un archivo que se quiere usar no tiene sentido volver a entrenar otro.
while ( loaded.lower() != 'y' and loaded.lower() != 'n'):
	loaded = input("Incorrect, try again.\n")

if ( loaded.lower() == 'y'):
	filepath = input("Name of the file containing the trained NN data.\n")
	model = RNG.Net()
	load_model(model, filepath)
	predict_new_election(model)	#Se pasa el control a la función que predice elecciones en base a un red entrenada
	sys.exit()

#path_arch = input("Enter the filepath to the candidate data:\n")
mycandidate = open('/Users/macbookair/Documents/Machine_learning/Eleccion/with_backprop/Data.csv','r')	
while (mycandidate.closed):
	path_arch = input('Incorrect, try again.\n')
	mycandidate = open(path_arch, 'r')

#path_arch = input("Enter the filepath to the economic data:\n")
#Este archivo contendrá los datos de la economia que son:
# desempleo el año de la eleccion y el anterior, pobreza el año de la eleccion y el anterior, 
# variación del pbi el año de la eleccion y el anterior,
# inflación el año de la eleccion y el anterior, nro de candidatos 
mycountry = open('/Users/macbookair/Documents/Machine_learning/Eleccion/with_backprop/Data_econom.csv','r')
while (mycountry.closed):
	path_arch = input('Incorrect, try again.\n')
	mycountry = open(path_arch, 'r')
	

training = Datos.Election( mycandidate, mycountry ) #función para inicializar el set de datos

mycandidate.close()
mycountry.close()

# Initialize model, loss function, and optimizer
model = RNG.Net()
criterion = nn.MSELoss()  # Quadratic error
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Gradient descent

# Training loop
epochs = 2500
prepared, votes = training.build_dataset()

print(prepared.shape)

for epoch in range(epochs):
	model.train()
	
	# Forward pass
	outputs = model(prepared)
	loss = criterion(outputs, votes)
	
	# Backward pass
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	if epoch % 10 == 0:
		print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


wishpred = input("Do you wish to predict an election? y/n:\n")
while ( wishpred.lower() != 'y' and wishpred.lower() != 'n'):
	wishpred = input('Incorrect, try again.\n')
	
if wishpred.lower() == 'y':
	predict_new_election(model)
	
if loaded !='y':
	wishsave = input("Do you wish to save the NN data? y/n:\n")
	while ( wishsave.lower() != 'y' and wishsave.lower() != 'n'):
		wishsave = input('Incorrect, try again.\n')
	
	if wishsave.lower() == 'y':
		save_model(model, "Trained_NN.pth")
	
	