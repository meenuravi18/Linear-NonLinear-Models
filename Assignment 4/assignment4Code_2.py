import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as sstats
from numpy import *
# Set some options for printing all the columns
pandas.set_option('precision', 7)

trainData = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 4/myeloma.csv')

trainData['Survival Time'] =trainData['Time']
trainData['Status'] = numpy.where((trainData['VStatus'])==0, 'Censored', 'Death')
nUnit = trainData.shape[0]
print(nUnit)


# # Calculate the Kaplan-Meier Product Limit Estimator for the Survival Function
xtab = pandas.crosstab(index = trainData['Survival Time'], columns = trainData['Status'])

lifeTable = pandas.DataFrame({'Survival Time': 0, 'Number Left': nUnit, 'Number of Events': 0, 'Number Censored': 0}, index = [0])
lifeTable = lifeTable.append(pandas.DataFrame({'Survival Time': xtab.index, 'Number of Events': xtab['Death'].to_numpy(),
                                               'Number Censored': xtab['Censored'].to_numpy()}),
                             ignore_index = True)

lifeTable[['Number at Risk']] = nUnit

nTime = lifeTable.shape[0]
probSurvival = 1.0
hazardFunction = 0.0
seProbSurvival = 0.0
lifeTable.at[0,'Prob Survival'] = probSurvival
lifeTable.at[0,'Prob Failure'] = 1.0 - probSurvival
lifeTable.at[0,'Cumulative Hazard'] = hazardFunction

for i in numpy.arange(1,nTime):
   nDeath = lifeTable.at[i,'Number of Events']
   nAtRisk = lifeTable.at[i-1,'Number Left'] - lifeTable.at[i-1,'Number Censored']
   nLeft = nAtRisk - nDeath
   probSurvival = probSurvival * (nLeft / nAtRisk)
   seProbSurvival = seProbSurvival + nDeath / nAtRisk / nLeft
   hazardFunction = hazardFunction + (nDeath / nAtRisk)
   lifeTable.at[i, 'SE Prob Survival'] = seProbSurvival
   lifeTable.at[i,'Number Left'] = nLeft
   lifeTable.at[i,'Number at Risk'] = nAtRisk
   lifeTable.at[i,'Prob Survival'] = probSurvival
   lifeTable.at[i,'Prob Failure'] = 1.0 - probSurvival
   lifeTable.at[i,'Cumulative Hazard'] = hazardFunction

lifeTable[['SE Prob Survival']] = lifeTable['Prob Survival'] * numpy.sqrt(lifeTable['SE Prob Survival'])
z25 = sstats.norm.ppf(0.975)
u = z25 * lifeTable['SE Prob Survival']
lifeTable[['Upper CI Prob Survival']] = lifeTable['Prob Survival'] + u
lifeTable[['Lower CI Prob Survival']] = lifeTable['Prob Survival'] - u

u = lifeTable['Upper CI Prob Survival']
lifeTable[['Upper CI Prob Survival']] = numpy.where(u > 1.0, 1.0, u)

u = lifeTable['Lower CI Prob Survival']
lifeTable[['Lower CI Prob Survival']] = numpy.where(u < 0.0, 0.0, u)





myeloma = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 4/myeloma.csv')

nUnit = myeloma.shape[0]
print(nUnit)
nUnit = 65



def SWEEPOperator (pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    
    A = numpy.copy(inputM)
    diagA = numpy.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = numpy.zeros(pDim)
            ANext[k, :] = numpy.zeros(pDim)
        A = ANext
    return (A, aliasParam, nonAliasParam)

fullX = (myeloma[['LogBUN']])
fullX = fullX.join(myeloma[['HGB']])
fullX = fullX.join(myeloma[['Platelet']])
fullX = fullX.join(myeloma[['Age']])
fullX = fullX.join(myeloma[['LogWBC']])
fullX = fullX.join(myeloma[['Frac']])
fullX = fullX.join(myeloma[['LogPBM']])
fullX = fullX.join(myeloma[['Protein']])
fullX = fullX.join(myeloma[['SCalc']])


fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)

print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])

modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
# print(modelX)
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)



#remove logwbc
fullX = myeloma[['LogBUN','HGB','Platelet','Age','Frac','LogPBM','Protein','SCalc']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

#remove platelet
fullX = myeloma[['LogBUN','HGB','Age','Frac','LogPBM','Protein','SCalc']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

#remove protein
fullX = myeloma[['LogBUN','HGB','Age','Frac','LogPBM','SCalc']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

#remove logpbm
fullX = myeloma[['LogBUN','HGB','Age','Frac','SCalc']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

#remove frac
fullX = myeloma[['LogBUN','HGB','Age','SCalc']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

#remove age
fullX = myeloma[['LogBUN','HGB','SCalc']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

#removescalc
fullX = myeloma[['LogBUN','HGB']]
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)
print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])
modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(myeloma[['Time','VStatus']])
cph = CoxPHFitter()
cph.fit(modelX, duration_col='Time', event_col='VStatus')
print(cph.params_)
cph.print_summary()
baseHazard = cph.baseline_hazard_
cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=False)

baseCumulativeHazard = cph.baseline_cumulative_hazard_


t = linspace(0, 108, 12)


plt.plot(lifeTable['Survival Time'], lifeTable['Cumulative Hazard'], marker = '+', markersize = 10, drawstyle = 'steps',label = 'Kaplan-Meier Estimator')
plt.plot(baseCumulativeHazard.index, baseCumulativeHazard, drawstyle = 'steps', marker = '+', label='Proportional Hazard model')
plt.xlabel('Survivial Time (Months)')
plt.ylabel('Cumulative Hazard Function')
plt.xticks(numpy.arange(0,96,12))
plt.grid(axis = 'both')
plt.legend()
plt.show()