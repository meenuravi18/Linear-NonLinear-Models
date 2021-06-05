import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as sstats

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
print(lifeTable)
lifeTable.to_csv("lifetable.csv")
plt.plot(lifeTable['Survival Time'], lifeTable['Prob Survival'], marker = '+', markersize = 10, drawstyle = 'steps')
plt.plot(lifeTable['Survival Time'], lifeTable['Upper CI Prob Survival'], marker = '+', markersize = 10, drawstyle = 'steps',
         linestyle = 'dashed', label = 'Upper Confidence Limit')
plt.plot(lifeTable['Survival Time'], lifeTable['Lower CI Prob Survival'], marker = '+', markersize = 10, drawstyle = 'steps',
         linestyle = 'dashed', label = 'Lower Confidence Limit')
plt.fill_between(lifeTable['Survival Time'],lifeTable['Lower CI Prob Survival'] ,lifeTable['Prob Survival'] , color='g', alpha=.3)
plt.fill_between(lifeTable['Survival Time'],lifeTable['Prob Survival'] , lifeTable['Upper CI Prob Survival'], color='y', alpha=.3)
plt.xlabel('Surivial Time (Months)')
plt.ylabel('Survival Function')
plt.xticks(numpy.arange(0,96,12))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend()
plt.show()

plt.plot(lifeTable['Survival Time'], lifeTable['Cumulative Hazard'], marker = '+', markersize = 10, drawstyle = 'steps')
plt.xlabel('Survivial Time (Months)')
plt.ylabel('Cumulative Hazard Function')
plt.xticks(numpy.arange(0,96,12))
# plt.yticks(numpy.arange(0.0,0.8,0.1))
plt.grid(axis = 'both')
plt.show()