import matplotlib.pyplot as plt
import numpy
import pandas

# Define a function to visualize the odds of a binary (0,1) target variable by nominal predictors
def TargetOddsByOneNominal (
   binary_target,       # target variable
   nom_predictor):      # nominal predictor

   countTable = pandas.crosstab(index = nom_predictor,
                                columns = binary_target, margins = False, dropna = True)
   oddsTable = countTable[1] / countTable[0]

   plt.plot(oddsTable.index, oddsTable, marker = 'o')
   plt.ylabel('Odds')
   plt.xlabel(oddsTable.index.names[0])
   plt.grid(axis = 'both', linestyle = '--', linewidth = 0.5)
   plt.show()   
   return (countTable, oddsTable)

def TargetOddsByTwoNominal (
   binary_target,       # target variable
   nom_predictor1,      # nominal predictor 1
   nom_predictor2):     # nominal predictor 2

   countTable = pandas.crosstab(index = [nom_predictor1, nom_predictor2],
                                columns = binary_target, margins = False, dropna = True)
   oddsTable = countTable[1] / countTable[0]

   plotData = oddsTable.unstack(level = 1)
   nom_value2 = plotData.columns
   for value in nom_value2:
      plt.plot(plotData.index, plotData[value], marker = 'o', label = value)
   plt.legend(title = oddsTable.index.names[1])
   plt.ylabel('Odds')
   plt.xlabel(oddsTable.index.names[0])
   plt.grid(axis = 'both', linestyle = '--', linewidth = 0.5)
   plt.show()   
   return (countTable, oddsTable)

#HO_claim_history = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/HO_claim_history.csv', delimiter = ',')
claim_history = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/claim_history.csv', delimiter = ',',
  usecols=['MSTATUS', 'CAR_TYPE', 'REVOKED', 'URBANICITY','CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME','CLM_COUNT','EXPOSURE'])
claim_history=claim_history.dropna().reset_index(drop=True)
catlist = ['MSTATUS', 'CAR_TYPE', 'REVOKED', 'URBANICITY','CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME','CLM_COUNT','EXPOSURE']
claim_history['freq']=claim_history['CLM_COUNT']/claim_history['EXPOSURE']
claim_history[['target']] = numpy.where(claim_history[['freq']] > 1, 1, 0)


varlist = ['MSTATUS', 'CAR_TYPE', 'REVOKED', 'URBANICITY']
varlist2=['CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME' ]
#Question 1
for v1 in varlist:
   uvalue, ucount = numpy.unique(claim_history[v1], return_counts = True)

   countTable, oddsTable = TargetOddsByOneNominal(binary_target = claim_history['target'],
                                      nom_predictor = claim_history[v1])
   oddsRatio = numpy.max(oddsTable) / numpy.min(oddsTable)
   print('Predictor = ', v1)
   print('Ration of Maximum Odds to Minimum Odds = ', oddsRatio)

for v1 in varlist2:
   uvalue, ucount = numpy.unique(claim_history[v1], return_counts = True)

   countTable, oddsTable = TargetOddsByOneNominal(binary_target = claim_history['target'],
                                      nom_predictor = claim_history[v1])
   
   oddsRatio = numpy.max(oddsTable) / numpy.min(oddsTable)
   print('Predictor = ', v1)
   print('Ration of Maximum Odds to Minimum Odds = ', oddsRatio)
