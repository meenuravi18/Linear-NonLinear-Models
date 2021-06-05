import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.special as sspecial
import scipy.stats as sstats

# Set some options for printing all the columns
pandas.set_option('display.max_columns', None)  
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.set_option('precision', 13)

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

# Use Bisection method to solve this equation for alpha: log(alpha) - digamma(alpha) = c, c is a positive scalar
def solve4Alpha (c, maxIter = 100, epsilon = 1e-10):

   # Find a0 such that f0 is greater than or equal to c
   a0 = 0.5
   while True:
      f0 = numpy.log(a0) - sspecial.digamma(a0)
      if (f0 < c):
         a0 = a0 / 2.0
      else:
         break

   # Find a1 such that f1 is less than or equal to c
   a1 = 2.0
   while True:
      f1 = numpy.log(a1) - sspecial.digamma(a1)
      if (f1 > c):
         a1 = a1 * 2.0
      else:
         break

   # Update the end-points
   for nIter in range(maxIter):
      alpha = (a0 + a1) / 2.0
      func = numpy.log(alpha) - sspecial.digamma(alpha)
      if (abs(func-c) > epsilon):
         if (func > c):
            a0 = alpha
         else:
            a1 = alpha
      else:
         break

   return (alpha)

# X is the Design matrix, a pandas dataframe
# y is the Severity vector, a pandas series
def GLM4Severity (X, y, maxIter = 20, maxStep = 5, tolLLK = 1e-3, tolBeta = 1e-10):
   nObs = len(y)
   yvec = y.to_numpy(dtype = float)
   ylog = numpy.log(yvec)

   # Generate the design matrix with the Intercept term as the first column
   if X is not None:
      Xp1 = X.copy()
      Xp1.insert(0, '_Intercept', 1.0)
   else:
      Xp1 = pandas.DataFrame(numpy.full((nObs,1), 1.0), columns = ['_Intercept'] )

   pDim = Xp1.shape[1]
   designX = Xp1.values
   designXT = numpy.transpose(designX)
   
   # Find the non-aliased columns
   inputM = numpy.dot(designXT, designX)
   outputM, aliasParam, nonAliasParam = SWEEPOperator (pDim, inputM, tol = 1e-10)

   # Make all objects as numpy
   designX = Xp1.values[:,nonAliasParam]
   designXT = numpy.transpose(designX)

   # Initialize beta array
   beta = numpy.zeros((len(nonAliasParam)))
   beta[0] = numpy.log(numpy.mean(y))
   nu = designX.dot(beta)
   y_pred = numpy.exp(nu)
   rvec = yvec / y_pred
   c = numpy.mean(rvec - numpy.log(rvec)) - 1.0
   alpha = solve4Alpha(c)
   uvec = - alpha * (y / y_pred + numpy.log(y_pred)) + (alpha - 1.0) * ylog   
   llk = numpy.sum(uvec) + nObs * (alpha * numpy.log(alpha) - sspecial.gammaln(alpha)) 

   # Prepare the iteration history table
   itList = [0, llk, alpha]
   for b in beta:
      itList.append(b)
   iterTable = [itList]

   for it in range(maxIter):
      rvec = yvec / y_pred
      gradient = numpy.dot(designXT, (rvec - 1.0))
      hessian = - numpy.dot(designXT, (rvec.reshape((nObs,1)) * designX))
      delta = numpy.linalg.solve(hessian, gradient)
      step = 1.0
      for iStep in range(maxStep):
         beta_next = beta - step * delta
         nu_next = numpy.dot(designX, beta_next)
         y_pred_next = numpy.exp(nu_next)
         rvec = yvec / y_pred_next
         c = numpy.mean(rvec - numpy.log(rvec)) - 1.0
         alpha_next = solve4Alpha(c)
         uvec = - alpha_next * (y / y_pred_next + numpy.log(y_pred_next)) + (alpha_next - 1.0) * ylog   
         llk_next = numpy.sum(uvec) + nObs * (alpha_next * numpy.log(alpha_next) - sspecial.gammaln(alpha_next)) 
         if ((llk_next - llk) > - tolLLK):
            break
         else:
            step = 0.5 * step
      diffBeta = beta_next - beta
      llk = llk_next
      beta = beta_next
      alpha = alpha_next
      y_pred = y_pred_next
      itList = [it+1, llk, alpha]
      for b in beta:
         itList.append(b)
      iterTable.append(itList)
      if (numpy.linalg.norm(diffBeta) < tolBeta):
         break

   rvec = yvec / y_pred
   hessian = numpy.dot(designXT, (rvec.reshape((nObs,1)) * designX))
   covBeta = numpy.linalg.inv(hessian) / alpha

   return(iterTable, nonAliasParam, llk, alpha, beta, covBeta, y_pred)

inputData = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/claim_history.csv',
                            delimiter = ',',
                            usecols = ['HOMEKIDS', 'KIDSDRIV', 'CAR_TYPE', 'CAR_AGE', 'CLM_AMT', 'CLM_COUNT'])

trainData = inputData[inputData['CLM_COUNT'] > 0.0].dropna()

y_train = trainData['CLM_AMT'] / trainData['CLM_COUNT']

# Summary statistics of Severity
print(y_train.describe())

# Histogram of Severity
plt.figure(figsize = (5,3), dpi = 200)
plt.hist(y_train, bins = numpy.arange(0,13000,500), fill = True, color = 'lightyellow', edgecolor = 'black')
plt.xlabel('Severity')
plt.ylabel('Number of Observations')
plt.xticks(numpy.arange(0,14000,2000))
plt.yticks(numpy.arange(0,600,50))
plt.grid(axis = 'y', linewidth = 0.7, linestyle = 'dashed')
plt.show()

# Intercept only model
iterTable, nonAliasParam, llk0, alpha, beta, covBeta, y_pred = GLM4Severity (None, y_train)
outTable = pandas.DataFrame(iterTable, columns = ['Iteration', 'Log-Likelihood', 'Scale', 'Intercept'])
print('Iteration History Table:\n', outTable)
print('\n')
df0 = len(nonAliasParam)

# Enter the first predictor
candidate_name = ['HOMEKIDS', 'KIDSDRIV', 'CAR_TYPE', 'CAR_AGE']
candidate_measure = ['interval', 'interval', 'categorical', 'interval']

for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      X_train = pandas.get_dummies(trainData[[X_name]].astype('category'))
   else:
      X_train = trainData[[X_name]]
   iterTable, nonAliasParam, llk, alpha, beta, covBeta, y_pred = GLM4Severity (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]
   outTable = pandas.DataFrame(iterTable, columns = ['Iteration', 'Log-Likelihood', 'Scale'] + param_name)

   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name), '\n')
   print('Iteration History Table:\n', outTable)
   print('\n')
   print('Deviance Test')
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   
# Enter the second predictor
X0_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category'))
iterTable, nonAliasParam, llk0, alpha, beta, covBeta, y_pred = GLM4Severity (X0_train, y_train)
df0 = len(nonAliasParam)

candidate_name = ['HOMEKIDS', 'KIDSDRIV', 'CAR_AGE']
for X_name in candidate_name:
   X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, alpha, beta, covBeta, y_pred = GLM4Severity (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]
   outTable = pandas.DataFrame(iterTable, columns = ['Iteration', 'Log-Likelihood', 'Scale'] + param_name)

   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name), '\n')
   print('Iteration History Table:\n', outTable)
   print('\n')
   print('Deviance Test')
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   
# Enter the third predictor
X0_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category'))
X0_train = X0_train.join(trainData[['KIDSDRIV']])
iterTable, nonAliasParam, llk0, alpha, beta, covBeta, y_pred = GLM4Severity (X0_train, y_train)
df0 = len(nonAliasParam)

candidate_name = ['HOMEKIDS', 'CAR_AGE']
for X_name in candidate_name:
   X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, alpha, beta, covBeta, y_pred = GLM4Severity (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]
   outTable = pandas.DataFrame(iterTable, columns = ['Iteration', 'Log-Likelihood', 'Scale'] + param_name)

   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name), '\n')
   print('Iteration History Table:\n', outTable)
   print('\n')
   print('Deviance Test')
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)

# Enter the fourth predictor
X0_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category'))
X0_train = X0_train.join(trainData[['KIDSDRIV', 'CAR_AGE']])
iterTable, nonAliasParam, llk0, alpha, beta, covBeta, y_pred = GLM4Severity (X0_train, y_train)
df0 = len(nonAliasParam)

candidate_name = ['HOMEKIDS']
for X_name in candidate_name:
   X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, alpha, beta, covBeta, y_pred = GLM4Severity (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]
   outTable = pandas.DataFrame(iterTable, columns = ['Iteration', 'Log-Likelihood', 'Scale'] + param_name)

   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name), '\n')
   print('Iteration History Table:\n', outTable)
   print('\n')
   print('Deviance Test')
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)

# Final model
X_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category'))
X_train = X_train.join(trainData[['KIDSDRIV', 'CAR_AGE']])
iterTable, nonAliasParam, llk, alpha, beta, covBeta, y_pred = GLM4Severity (X_train, y_train)
X_column = ['Intercept'] + X_train.columns.to_list()
param_name = [X_column[i] for i in nonAliasParam]

# Final parameter estimates
print('Scale Parameter = ', alpha)
outCoefficient = pandas.Series(beta, index = param_name)
print('Parameter Estimates:\n', outCoefficient, numpy.exp(outCoefficient))

# Final correlation matrix
outCovb = pandas.DataFrame(covBeta, index = param_name, columns = param_name)
stddev = numpy.sqrt(numpy.diag(outCovb))
outCorrb = outCovb / numpy.outer(stddev, stddev)
print('Correlation Matrix of Parameter Estimates:\n', outCorrb)

plt.figure(figsize = (7,4), dpi = 200)
plt.scatter(y_train,y_pred, marker = 'o', c = trainData['CLM_COUNT'], s = 20)
plt.colorbar().set_label('Number of Claims')
plt.xlabel('Observed Severity')
plt.ylabel('Predicted Severity')
plt.xticks(numpy.arange(0,160000,20000))
plt.yticks(numpy.arange(0,5000,500))
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()

# Zoom into the graph
ythresh = 20000
yy = y_train[y_train <= ythresh]
py = y_pred[y_train <= ythresh]
cc = trainData['CLM_COUNT']
cc = cc[y_train <= ythresh]
plt.figure(figsize = (7,4), dpi = 200)
plt.scatter(yy,py, marker = 'o', c = cc, s = 20)
plt.colorbar().set_label('Number of Claims')
plt.xlabel('Observed Severity')
plt.ylabel('Predicted Severity')
plt.xticks(numpy.arange(0,25000,5000))
plt.yticks(numpy.arange(0,5000,500))
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()