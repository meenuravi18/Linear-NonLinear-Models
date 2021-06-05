import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as sstats

# Set some options for printing all the columns
pandas.set_option('precision', 7)

whas500 = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 4/whas500.csv')

nUnit = 500

# Distribution of Status
# statusDistribution = whas500.groupby('fstat').size()
# plt.bar(statusDistribution.index, statusDistribution)
# plt.xlabel('Status')
# plt.ylabel('Number of Respondents')
# plt.xticks(range(2))
# plt.yticks(range(0,320,20))
# plt.grid(axis = 'y')
# plt.show()

# Calculate the Kaplan-Meier Product Limit Estimator for the Survival Function
# xtab = pandas.crosstab(index = whas500['lenfol'], columns = whas500['fstat'])

# lifeTable = pandas.DataFrame({'Survival Time': 0, 'Number Left': nUnit, 'Number of Events': 0, 'Number Censored': 0}, index = [0])
# lifeTable = lifeTable.append(pandas.DataFrame({'Survival Time': xtab.index, 'Number of Events': xtab[1].to_numpy(),
#                                                'Number Censored': xtab[0].to_numpy()}),
#                              ignore_index = True)

# lifeTable[['Number at Risk']] = nUnit

# nTime = lifeTable.shape[0]
# probSurvival = 1.0
# hazardFunction = 0.0
# seProbSurvival = 0.0
# lifeTable.at[0,'Prob Survival'] = probSurvival
# lifeTable.at[0,'Prob Failure'] = 1.0 - probSurvival
# lifeTable.at[0,'Cumulative Hazard'] = hazardFunction

# for i in numpy.arange(1,nTime):
#    nDeath = lifeTable.at[i,'Number of Events']
#    nAtRisk = lifeTable.at[i-1,'Number Left'] - lifeTable.at[i-1,'Number Censored']
#    nLeft = nAtRisk - nDeath
#    probSurvival = probSurvival * (nLeft / nAtRisk)
#    seProbSurvival = seProbSurvival + nDeath / nAtRisk / nLeft
#    hazardFunction = hazardFunction + (nDeath / nAtRisk)
#    lifeTable.at[i, 'SE Prob Survival'] = seProbSurvival
#    lifeTable.at[i,'Number Left'] = nLeft
#    lifeTable.at[i,'Number at Risk'] = nAtRisk
#    lifeTable.at[i,'Prob Survival'] = probSurvival
#    lifeTable.at[i,'Prob Failure'] = 1.0 - probSurvival
#    lifeTable.at[i,'Cumulative Hazard'] = hazardFunction

# lifeTable[['SE Prob Survival']] = lifeTable['Prob Survival'] * numpy.sqrt(lifeTable['SE Prob Survival'])
# z25 = sstats.norm.ppf(0.975)
# u = z25 * lifeTable['SE Prob Survival']
# lifeTable[['Upper CI Prob Survival']] = lifeTable['Prob Survival'] + u
# lifeTable[['Lower CI Prob Survival']] = lifeTable['Prob Survival'] - u

# u = lifeTable['Upper CI Prob Survival']
# lifeTable[['Upper CI Prob Survival']] = numpy.where(u > 1.0, 1.0, u)

# u = lifeTable['Lower CI Prob Survival']
# lifeTable[['Lower CI Prob Survival']] = numpy.where(u < 0.0, 0.0, u)

# plt.plot(lifeTable['Survival Time'], lifeTable['Prob Survival'], drawstyle = 'steps')
# plt.plot(lifeTable['Survival Time'], lifeTable['Upper CI Prob Survival'], drawstyle = 'steps',
#          linestyle = 'dashed', label = 'Upper Confidence Limit')
# plt.plot(lifeTable['Survival Time'], lifeTable['Lower CI Prob Survival'], drawstyle = 'steps',
#          linestyle = 'dashed', label = 'Lower Confidence Limit')
# plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admissio to Date of Last Follow-up)')
# plt.ylabel('Survival Function')
# plt.xticks(numpy.arange(0,2920,365))
# plt.yticks(numpy.arange(0.0,1.1,0.1))
# plt.grid(axis = 'both')
# plt.legend()
# plt.show()

# plt.plot(lifeTable['Survival Time'], lifeTable['Cumulative Hazard'], drawstyle = 'steps')
# plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admissio to Date of Last Follow-up)')
# plt.ylabel('Cumulative Hazard Function')
# plt.xticks(numpy.arange(0,2920,365))
# plt.yticks(numpy.arange(0.0,3.5,0.5))
# plt.grid(axis = 'both')
# plt.show()

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

# Gender + Age
fullX = pandas.get_dummies(whas500[['gender']].astype('category'))
fullX = fullX.join(whas500[['age']])
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)

print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])

modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
print(modelX)
# modelX = modelX.join(whas500[['lenfol','fstat']])

# from lifelines import CoxPHFitter

# cph = CoxPHFitter()
# cph.fit(modelX, duration_col='lenfol', event_col='fstat')
# print(cph.params_)
# print("=============================")
# cph.print_summary()

# baseHazard = cph.baseline_hazard_

# plt.plot(baseHazard.index, baseHazard, drawstyle = 'steps', marker = '+')
# plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admissio to Date of Last Follow-up)')
# plt.ylabel('Baseline Hazard Function')
# plt.xticks(numpy.arange(0,2920,365))
# plt.yticks(numpy.arange(0.0,5.0,0.5))
# plt.grid(axis = 'both')
# plt.show()

# cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=True)

# # age + sysbp + diasbp + bmi
# fullX = whas500[['age','sysbp','diasbp','bmi']]
# fullX.insert(0, '_Intercept', 1.0)
# XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
# pDim = XtX.shape[0]
# invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim, XtX, 1.0e-8)

# print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])

# modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
# modelX = modelX.join(whas500[['lenfol','fstat']])

# cph = CoxPHFitter()
# cph.fit(modelX, duration_col='lenfol', event_col='fstat')
# print(cph.params_)
# cph.print_summary()

# baseHazard = cph.baseline_hazard_

# plt.plot(baseHazard.index, baseHazard, drawstyle = 'steps', marker = '+')
# plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admissio to Date of Last Follow-up)')
# plt.ylabel('Baseline Hazard Function')
# plt.xticks(numpy.arange(0,2920,365))
# plt.yticks(numpy.arange(0.0,5.0,0.5))
# plt.grid(axis = 'both')
# plt.show()

# cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=True)