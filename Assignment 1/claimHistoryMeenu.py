import matplotlib.pyplot as plt
import numpy
import pandas
# import scipy.special
#import scipy
from scipy import special


pandas.set_option('precision', 7)

inputData = pandas.read_csv('claim_history.csv', delimiter = ',', usecols = ['AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS',
	'TIF','TRAVTIME','YOJ', 'CLM_COUNT'])
#inputData = pandas.read_csv('claim_history.csv', delimiter = ',', usecols = ['HOMEKIDS', 'KIDSDRIV','CLM_COUNT'])
inputData = inputData.dropna()
'''
print('CLM_COUNT Frequency Table:/n', inputData.groupby('CLM_COUNT').size())
print('AGE Frequency Table:/n', inputData.groupby('AGE').size())
print('CAR_AGE Frequency Table:/n', inputData.groupby('CAR_AGE').size())
print('CAR_AGE Frequency Table:/n', inputData.groupby('CAR_AGE').size())
print('HOMEKIDS Frequency Table:/n', inputData.groupby('HOMEKIDS').size())
print('KIDSDRIV Frequency Table:/n', inputData.groupby('KIDSDRIV').size())
print('MVR_PTS Frequency Table:/n', inputData.groupby('MVR_PTS').size())
print('TIF Frequency Table:/n', inputData.groupby('TIF').size())
print('TRAVTIME Frequency Table:/n', inputData.groupby('TRAVTIME').size())
print('YOJ Frequency Table:/n', inputData.groupby('YOJ').size())
print('CLM_COUNT Frequency Table:/n', inputData.groupby('CLM_COUNT').size())
'''
# # Target histogram
# plt.hist(inputData['CLM_COUNT'], bins = range(10), align = 'left')
# plt.xlabel('CLM_COUNT')
# plt.ylabel('Frequency')
# plt.xticks(range(10))
# plt.grid(axis = 'y')
# plt.show()

# # HOMEKIDS histogram
# plt.hist(inputData['HOMEKIDS'], bins = range(6), align = 'left')
# plt.xlabel('HOMEKIDS')
# plt.ylabel('Frequency')
# plt.xticks(range(6))
# plt.grid(axis = 'y')
# plt.show()

# # KIDSDRIV histogram
# plt.hist(inputData['KIDSDRIV'], bins = range(5), align = 'left')
# plt.xlabel('KIDSDRIV')
# plt.ylabel('Frequency')
# plt.xticks(range(5))
# plt.grid(axis = 'y')
# plt.show()

#X_train = inputData[['HOMEKIDS', 'KIDSDRIV']]

X_train = inputData[['AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS',
	'TIF','TRAVTIME','YOJ']]
y_train = inputData['CLM_COUNT']

nObs = len(y_train)

# Precompute the ln(y!)
constLLK = special.loggamma(y_train+1.0)

# Generate the design matrix with the Intercept term as the first column
Xp1 = X_train.copy()
Xp1.insert(0, '_Intercept', 1.0)
nParameter = Xp1.shape[1]


# Make all objects as numpy
designX = Xp1.values
designXT = numpy.transpose(designX)


# Specify some constants
maxIter = 50
maxStep = 5
tolLLK = 1e-3
tolBeta = 1e-10

# Initialize beta array
beta = numpy.array([numpy.log(numpy.mean(y_train)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
nu = designX.dot(beta)
y_pred = numpy.exp(nu)
llk = numpy.sum(y_train * nu - y_pred - constLLK)

# Prepare the iteration history table
itList = [0, llk]
for b in beta:
   itList.append(b)
iterTable = [itList]

for it in range(maxIter):
   gradient = numpy.dot(designXT, (y_train - y_pred))
   hessian = - numpy.dot(designXT, (y_pred.reshape((nObs,1)) * designX))
   delta = numpy.linalg.solve(hessian, gradient)
   step = 1.0
   for iStep in range(maxStep):
      beta_next = beta - step * delta
      nu_next = numpy.dot(designX, beta_next)
      y_pred_next = numpy.exp(nu_next)
      llk_next = numpy.sum(y_train * nu_next - y_pred_next - constLLK)
      if ((llk_next - llk) > - tolLLK):
         break
      else:
         step = 0.5 * step
   diffBeta = beta_next - beta
   llk = llk_next
   beta = beta_next
   y_pred = y_pred_next
   itList = [it+1, llk]
   for b in beta:
      itList.append(b)
   iterTable.append(itList)
   if (numpy.linalg.norm(diffBeta) < tolBeta):
      break
outTable = pandas.DataFrame(iterTable, columns = ['ITERATION','LOG LIKELIHOOD','Intercept', 'AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS',
	'TIF','TRAVTIME','YOJ'])

#outTable = pandas.DataFrame(iterTable, columns = ['AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS',
#	'TIF','TRAVTIME','YOJ'])
print('Iteration History Table:\n', outTable)
#outTable.to_csv("iterationTable.csv",index=False,header=True,float_format='%.7f')

# Final parameter estimates
outCoefficient = pandas.Series(beta, index = ['Intercept', 'AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS',
	'TIF','TRAVTIME','YOJ'])
estimates=outCoefficient.tolist()

#outCoefficient.to_csv("parameters.csv",index=False,header=True,float_format='%.7f')
# Final covariance matrix
hessian = - numpy.dot(designXT, (y_pred.reshape((nObs,1)) * designX))
outCovb = pandas.DataFrame(numpy.linalg.inv(- hessian),
                            index = ['Intercept', 'AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS','TIF','TRAVTIME','YOJ'],
                            columns = ['Intercept', 'AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS','TIF','TRAVTIME','YOJ'])

#outCovb.to_csv("covariance.csv",index=False,header=True,float_format='%.7f')
cols = ['Intercept', 'AGE','CAR_AGE','HOMEKIDS', 'KIDSDRIV', 'MVR_PTS','TIF','TRAVTIME','YOJ']
SE=[]
    
# # From covariance matrix to correlation matrix (did not handle zero standard deviations)
stddev = numpy.sqrt(numpy.diag(outCovb))
for i in range(0,len(stddev)):
    SE.append([cols[i],stddev[i]])
Stderr = pandas.DataFrame.from_records(SE)
#Stderr.to_csv("standardError.csv",index=False,header=True,float_format='%.7f')
outCorrb = outCovb / numpy.outer(stddev, stddev)
outCorrb.insert(0,'column',cols)
#outCorrb.to_csv("correlation.csv",index=False,header=True,float_format='%.7f')

copy=Stderr.copy()
copy.columns = ["columns",'SE']
copy["estimates"]=estimates

copy['lowerlimit'] = copy['estimates']-(1.96*copy['SE'])
copy['upperlimit'] = copy['estimates']+(1.96*copy['SE'])
copy.to_csv("cinfidenceInterval.csv",index=False,header=True,float_format='%.7f')


