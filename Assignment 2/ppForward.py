import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.linear_model as lm
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

# X is the Design matrix, a pandas dataframe
# y is the Severity vector, a pandas series
def GLM4PurePremium (X, y, exposure, tweedieP = 1.5, maxIter = 50, maxStep = 5, tolLLK = 1e-3, tolBeta = 1e-10):
   nObs = len(y)
   yvec = y.to_numpy(dtype = float)
   ovec = numpy.log(exposure.to_numpy(dtype = float))
   two_p = 2.0 - tweedieP
   one_p = 1.0 - tweedieP
   ypow21 = numpy.power(y, two_p) / two_p / one_p 

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
   beta[0] = numpy.log(numpy.mean(yvec))
   nu = ovec + designX.dot(beta)
   y_pred = numpy.exp(nu)
   powvec = numpy.power(y_pred, one_p)
   devvec = 2.0 * (ypow21 - yvec * powvec / one_p + y_pred * powvec / two_p)
   qllk = - numpy.sum(devvec) / 2.0

   # Prepare the iteration history table
   itList = [0, qllk]
   for b in beta:
      itList.append(b)
   iterTable = [itList]

   for it in range(maxIter):
      rvec = (y_pred - yvec) * powvec
      gradient = 2.0 * numpy.dot(designXT, rvec)
      svec = (two_p * y_pred - one_p * yvec) * powvec
      hessian = 2.0 * numpy.dot(designXT, (svec.reshape((nObs,1)) * designX))
      delta = numpy.linalg.solve(hessian, gradient)
      step = 1.0
      for iStep in range(maxStep):
         beta_next = beta - step * delta
         nu_next = ovec + numpy.dot(designX, beta_next)
         y_pred_next = numpy.exp(nu_next)
         powvec_next = numpy.power(y_pred_next, one_p)
         devvec = 2.0 * (ypow21 - yvec * powvec_next / one_p + y_pred_next * powvec_next / two_p)
         qllk_next = - numpy.sum(devvec) / 2.0
         if ((qllk_next - qllk) > - tolLLK):
            break
         else:
            step = 0.5 * step
      diffBeta = beta_next - beta
      qllk = qllk_next
      beta = beta_next
      powvec = powvec_next
      y_pred = y_pred_next
      itList = [it+1, qllk]
      for b in beta:
         itList.append(b)
      iterTable.append(itList)
      if (numpy.linalg.norm(diffBeta) < tolBeta):
         break

   svec = ((two_p * y_pred - one_p * yvec) * powvec)
   hessian = 2.0 * numpy.dot(designXT, (svec.reshape((nObs,1)) * designX))

   devvec = (ypow21 - yvec * powvec / one_p + y_pred * powvec / two_p)
   phi = numpy.sum(devvec) / (nObs - len(nonAliasParam))
   covBeta = numpy.linalg.inv(hessian) / phi

   return(iterTable, nonAliasParam, qllk, phi, beta, covBeta, y_pred)

catName = ['CAR_TYPE']
intName = ['CAR_AGE', 'HOMEKIDS', 'KIDSDRIV']
yName = 'CLM_AMT'
eName = 'EXPOSURE'

inputData = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/claim_history.csv',
                            delimiter = ',',
                            usecols = [yName] + [eName] + catName + intName)

trainData = inputData[inputData[eName] > 0.0].dropna()

y_train = trainData[yName]
e_train = trainData[eName]

# Summary statistics of Total Amount of Claims
print(y_train.describe())

# Histogram of CLM_AMT
plt.hist(y_train, bins = numpy.arange(0,13000,500), fill = True, color = 'lightyellow', edgecolor = 'black')
plt.xlabel(yName)
plt.ylabel('Number of Observations')
plt.xticks(numpy.arange(0,14000,2000))
plt.grid(axis = 'y', linewidth = 0.7, linestyle = 'dashed')
plt.show()

# Estimate the Tweedie's P value
xtab = pandas.pivot_table (trainData, index = ['CAR_TYPE','CAR_AGE','HOMEKIDS','KIDSDRIV'],
                           values = ['CLM_AMT'], aggfunc = ['count', 'mean', 'var'])
xtab = xtab.dropna()

xtab['lnMean'] = numpy.where(xtab['mean'] > 0.0, numpy.log(xtab['mean']), numpy.NaN)
xtab['lnVariance'] = numpy.where(xtab['var'] > 0.0, numpy.log(xtab['var']), numpy.NaN)
xtab = xtab.dropna()

# Scatterplot of lnVariance vs lnMean
plt.figure(figsize = (10,6), dpi = 200)
plt.scatter(xtab['lnMean'], xtab['lnVariance'])
plt.xlabel('Log(Mean)')
plt.ylabel('Log(Variance)')
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()

linReg = lm.LinearRegression(fit_intercept = True)
linRegModel = linReg.fit(xtab[['lnMean']], xtab['lnVariance'])

tweedieP = linRegModel.coef_
scalePhi = numpy.exp(linRegModel.intercept_)

# Begin Forward Selection

X0_train = None

nPredictor = len(catName) + len(intName)
stepSummary = pandas.DataFrame(index = range(nPredictor+1),
                               columns = ['Predictor', 'ModelDF', 'QuasiLLK', 'Scale', 'Deviance', 'DevDF', 'DevSig'])

# Intercept only model
iterTable, nonAliasParam, qllk0, phi0, beta, covBeta, y_pred = GLM4PurePremium (None, y_train, e_train, tweedieP = tweedieP)
df0 = len(nonAliasParam)
stepSummary.iloc[0] = ['Intercept', df0, qllk0, phi0, numpy.NaN, numpy.NaN, numpy.NaN]

cName = catName.copy()
iName = intName.copy()
entryThreshold = 0.05

for step in range(nPredictor):
   enterName = ''
   stepName = numpy.empty((0), dtype = str)
   stepStats = numpy.empty((0,6), dtype = float)

   # Enter the next predictor
   for X_name in cName:
      X_train = pandas.get_dummies(trainData[[X_name]].astype('category'))
      if (X0_train is not None):
         X_train = X0_train.join(X_train)
      iterTable, nonAliasParam, qllk, phi, beta, covBeta, y_pred = GLM4PurePremium (X_train, y_train, e_train, tweedieP = tweedieP)
      nParameter = len(nonAliasParam)
      devChiSq = 2.0 * (qllk - qllk0) / phi0
      devDF = nParameter - df0
      devPValue = sstats.chi2.sf(devChiSq, devDF)
      stepName = numpy.append(stepName, numpy.array([X_name]), axis = 0)
      stepStats = numpy.append(stepStats,
                               numpy.array([[nParameter, qllk, phi, devChiSq, devDF, devPValue]]),
                               axis = 0)
   for X_name in iName:
      X_train = trainData[[X_name]]
      if (X0_train is not None):
         X_train = X0_train.join(X_train)
      iterTable, nonAliasParam, qllk, phi, beta, covBeta, y_pred = GLM4PurePremium (X_train, y_train, e_train, tweedieP = tweedieP)
      nParameter = len(nonAliasParam)
      devChiSq = 2.0 * (qllk - qllk0) / phi0
      devDF = nParameter - df0
      devPValue = sstats.chi2.sf(devChiSq, devDF)
      stepName = numpy.append(stepName, numpy.array([X_name]), axis = 0)
      stepStats = numpy.append(stepStats,
                               numpy.array([[nParameter, qllk, phi, devChiSq, devDF, devPValue]]),
                               axis = 0)

   # Find a predictor to enter, if any
   minPValue = 1.1
   minI = -1
   for i in range(stepStats.shape[0]):
      thisPValue = stepStats[i,5] 
      if (thisPValue < minPValue):
         minPValue = thisPValue
         minI = i

   if (minPValue <= entryThreshold):
      enterName = stepName[minI]
      addList = [enterName]
      for v in stepStats[minI,:]:
         addList.append(v)
      stepSummary.loc[step+1] = addList
      df0 = stepStats[minI,0]
      qllk0 = stepStats[minI,1]
      phi0 = stepStats[minI,2]

      # Find the measurement level of enterName
      iCat = -1
      try:
         iCat = cName.index(enterName)
         X_train = pandas.get_dummies(trainData[[enterName]].astype('category'))
         if (X0_train is not None):
            X0_train = X0_train.join(X_train)
         else:
            X0_train = X_train
         cName.remove(enterName)
      except ValueError:
         iCat = -1

      if (iCat == -1):
         iInt = -1
         try:
            iInt = iName.index(enterName)
            X_train = trainData[[enterName]]
            if (X0_train is not None):
               X0_train = X0_train.join(X_train)
            else:
               X0_train = X_train
            iName.remove(enterName)
         except ValueError:
            iInt = -1
   else:
      break

   # Print debugging output
   print('Step = ', step+1)
   print('Step Statistics:')
   print(stepName)
   print(stepStats)
   print('Enter predictor = ', enterName)
   print('Minimum P-Value =', minPValue)
   print('\n')

# End of forward selection

# The final model is URBANICITY, CAR_TYPE, MVR_PTS, MSTATUS, TRAVTIME, CAR_AGE, REVOKED, and TIF.

predName = ['CAR_TYPE', 'CAR_AGE', 'HOMEKIDS', 'KIDSDRIV']
X0_train = None

for X_name in predName:
   try:
      iCat = catName.index(X_name)
   except ValueError:
      iCat = -1

   try:
      iInt = intName.index(X_name)
   except ValueError:
      iInt = -1

   if (iCat >= 0):
      X_train = pandas.get_dummies(trainData[[X_name]].astype('category'))
   elif (iInt >= 0):
      X_train = trainData[[X_name]]
   else:
      X_train = None

   if (X0_train is not None):
      X0_train = X0_train.join(X_train)
   else:
      X0_train = X_train

iterTable, nonAliasParam, qllk, phi, beta, covBeta, y_pred = GLM4PurePremium (X0_train, y_train, e_train, tweedieP = tweedieP)

# Assemble the complete parameter estimates
parameter = pandas.DataFrame(index = range(1+X0_train.shape[1]))
parameter['Name'] = X0_train.columns.insert(0,'Intercept')
parameter['DF'] = 0
parameter['DF'].iloc[nonAliasParam] = 1
parameter['Estimate'] = 0.0
parameter['Estimate'].iloc[nonAliasParam] = beta.copy()
parameter['Exp(Estimate)'] = numpy.exp(parameter['Estimate'])

print('Parameter Estimate')
print(parameter)

# Plot the predicted Number of Claims versus the observed Number of Claims
plt.figure(figsize = (10,6), dpi = 200)
plt.scatter(y_train,y_pred, marker = 'o', c = e_train, s = 20)
plt.colorbar().set_label('Exposure')
plt.xlabel('Observed ' + yName)
plt.ylabel('Predicted ' + yName)
plt.xticks(numpy.arange(0,160000,20000))
plt.yticks(numpy.arange(0,12000,1000))
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()
