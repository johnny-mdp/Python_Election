import pandas as pd
import torch

class Candidates():
	def __init__(self, path):
		"""
		Each candidate has a
		year, 
		ideology: a float (0 left to 1 right)
		power: a boolean (0 the candidate's party isn't in power, 1 if it is
		favouravility: float the candidate's favourability rating normalized to 1
		polling: float, the candidate's pollin normalized to 1
		"""


class Country():
	def __init__(self, path):
		"""
		The country economic data used is
		year,
		unemployment: float between 0 and 1 representing the unemployment in the election year
		prev_unemployment: float between 0 and 1 representing the unemployment in the previous year
		poverty: float between 0 and 1 representing the poverty rate in the election year
		prev_poverty: float between 0 and 1 representing the poverty rate in the previous year
		gdp_growth: float greater than -1 representing the GDP growth rate in the election year
		prev_gdp_growth: float greater than -1 representing the GDP growth rate in the previous year
		inflation: float greater than -1 representing the inflation  rate in the election year
		prev_inflation: float greater than -1 representing the inflation growth rate in the previous year
		num_cand: number of candidates on each election
		"""
		self.country = pd.read_csv(path)




import numpy as np

class Election():
	def __init__(self, candidatesfile, economyfile):		
		candidates_df = pd.read_csv(candidatesfile)
		self.candidates = candidates_df.iloc[:, :-1]  # All columns except result
		self.results = candidates_df.iloc[:, -1]		  # Just the result column
		self.country = pd.read_csv(economyfile)

	def build_dataset(self, country=None, candidates=None):
		data = []
		targets = []
		
			
		for i in range(len(self.candidates)):
			candidate_row = self.candidates.iloc[i]
			year = candidate_row["year"]

			# Get matching country row by year
			country_row = self.country[self.country["year"] == year]
			if country_row.empty:
				continue  # Skip if no match

			# Drop year columns to extract only features
			cand_features = candidate_row.drop("year").values.astype(float)
			econ_features = country_row.drop(columns=["year", "num_cand"]).values.flatten().astype(float)

			# Combine candidate and country data
			combined = np.concatenate([cand_features, econ_features])
			data.append(combined)

			# Get vote share for this candidate
			vote_share = self.results.iloc[i]
			targets.append(vote_share)
		
		print(len(data), len(data[0]))
		tensor_data = torch.tensor(data, dtype=torch.float32)
		tensor_targets = torch.tensor(targets, dtype=torch.float32)
		
		
		return tensor_data, tensor_targets




def SaveElection(argentina,candidates,myresults):
#Guarda los resultados de la simulacion en un archivo .txt
	argentina.to_csv(myresults, sep="\t", index=False)
	myresults.write('\n')
	candidates.to_csv(myresults, sep="\t", index=False)
	myresults.write('\n\n')

#termina guardar_eleccion


def ReadData():
	#Ingresa los datos de la economía del pais y nro decandidatos
	year = int(input( "Ingrese el año de la eleccion:\n"))
	unemp = float(input("Ingrese tasa de desempleo (normalizada a 1) el año de la eleccion:\n"))
	while ( unemp< 0 or unemp> 1):
		unemp = float(input("Valor incorrecto. Intente nuevamente.\n"))
	prev_unemp = float(input("Ingrese tasa de desempleo (normalizada a 1) el año anterior:\n"))
	while ( prev_unemp< 0 or prev_unemp> 1):
		prev_unemp = float(input("Valor incorrecto. Intente nuevamente.\n"))
	poverty = float(input("Ingrese tasa de pobreza (normalizada a 1) el año de la eleccion:\n"))
	while ( poverty< 0 or poverty> 1):
		poverty = float(input("Valor incorrecto. Intente nuevamente.\n"))
	prev_poverty = float(input("Ingrese tasa de pobreza (normalizada a 1) el año anterior:\n"))
	while ( prev_poverty< 0 or prev_poverty> 1):
		prev_poverty = float(input("Valor incorrecto. Intente nuevamente.\n"))
	gdp_growth = float(input("Ingrese tasa de crecimiento del PBI (normalizada a 1) el año de la eleccion:\n"))
	while ( gdp_growth< -1 ):
		gdp_growth = float(input("Valor incorrecto. Intente nuevamente.\n"))
	prev_gdp_growth = float(input("Ingrese tasa de crecimiento del PBI (normalizada a 1) el año anterior:\n"))
	while ( prev_gdp_growth< -1 ):
		prev_gdp_growth = float(input("Valor incorrecto. Intente nuevamente.\n"))
	inflation = float(input("Ingrese tasa de inflacion (normalizada a 1) el año de la eleccion:\n"))
	while ( inflation < -1 ):
		inflation = float(input("Valor incorrecto. Intente nuevamente.\n"))
	prev_inflation = float(input("Ingrese tasa de inflacion (normalizada a 1) el año anterior:\n"))
	while ( prev_inflation< -1 ):
		prev_inflation = float(input("Valor incorrecto. Intente nuevamente.\n"))
	cand_per_year = int(input("Por favor, ingrese el número de candidatos en la elección:\n"))
	argentina = pd.DataFrame({
    'year': [year],
    'unemployment': [unemp],
    'prev_unemployment': [prev_unemp],
    'poverty': [poverty],
    'prev_poverty': [prev_poverty],
    'gdp_growth': [gdp_growth],
    'prev_gdp_growth': [prev_gdp_growth],
    'inflation': [inflation],
    'prev_inflation': [prev_inflation],
    'num_cand': [cand_per_year]})
	
	candidates = pd.DataFrame(columns=[
		'year',
		'ideology',
		'power',
		'image',
		'polling'])
	
	for i in range(0, cand_per_year):
		#Ingresa los datos de cada candidato
		ideology = float(input("Ingrese el puntaje ideologico del candidato (0 izquierda, 1 derecha):\n"))
		while (ideology < 0 or ideology>1 ):
			ideology = float(input("Valor incorrecto. Intente nuevamente.\n"))
		power = int(input("¿El candidato forma parte del partido de gobierno? 1 SÍ, 0 NO:\n"))
		image= float(input("¿Cual es su imágen positiva + regular?\n"))
		while ( image < 0 or image > 1 ):
			image = float(input("Valor incorrecto. Intente nuevamente.\n"))
		polling = float(input("¿Cual es su intención de voto normalizada a 1?\n"))
		while ( polling < 0 or polling> 1 ):
			polling = float(input("Valor incorrecto. Intente nuevamente.\n"))
		candi  = pd.DataFrame([{
			'year': year,
			'ideology':ideology,
			'power':power,
			'image':image,
			'polling':polling}])
		candidates = pd.concat([candidates, candi], ignore_index=True)
	results = [0.0]*cand_per_year
	print(candidates, argentina)
	
	data = []
	for i in range(cand_per_year):
		candidate_row = candidates.iloc[i]
		year = candidate_row["year"]

		# Get matching country row by year
		country_row = argentina[argentina["year"] == year]
		if country_row.empty:
			continue  # Skip if no match

		# Drop year columns to extract only features
		cand_features = candidates.iloc[i].drop("year").values.astype(float)
		econ_features = argentina.drop(columns=["year", "num_cand"]).values.flatten().astype(float)

		# Combine candidate and country data
		combined = np.concatenate([cand_features, econ_features])
		data.append(combined)

	print(len(data), len(data[0]))
	tensor_data = torch.tensor(data, dtype=torch.float32)
	
	return tensor_data, argentina, candidates