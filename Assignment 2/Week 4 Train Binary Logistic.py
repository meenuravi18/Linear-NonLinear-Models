import matplotlib.pyplot as plt
import numpy
import pandas

pandas.set_option('precision', 7)

# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# The SWEEP Operator
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

# X is a pandas dataframe, y is a pandas series
def GLM4BinaryLogistic (X, y, maxIter = 20, maxStep = 5, tolLLK = 1e-3, tolBeta = 1e-10):
   nObs = len(y)

   # Generate the design matrix with the Intercept term as the first column
   if X is not None:
      Xp1 = pandas.DataFrame(numpy.full((nObs,1), 1.0), columns = ['_Intercept'], index = X.index)
      Xp1 = Xp1.join(X)
   else:
      Xp1 = pandas.DataFrame(numpy.full((nObs,1), 1.0), columns = ['_Intercept'])

   pDim = Xp1.shape[1]
   designX = Xp1.to_numpy()
   designXT = numpy.transpose(designX)

   # Find the non-aliased columns
   inputM = numpy.dot(designXT, designX)
   outputM, aliasParam, nonAliasParam = SWEEPOperator (pDim, inputM, tol = 1e-10)

   # Make all objects as numpy
   designX = Xp1.values[:,nonAliasParam]
   designXT = numpy.transpose(designX)

   # Initialize predicted probabilities
   pEvent = numpy.mean(y)
   pNonEvent = 1.0 - pEvent
   odds = pEvent / pNonEvent
   y_predProb = numpy.full(nObs, pEvent)  
   beta = numpy.zeros((len(nonAliasParam)))
   beta[0] = numpy.log(odds)
   llk = numpy.sum(y * beta[0] + numpy.log(pNonEvent))

   # Prepare the iteration history table
   itList = [0, llk]
   for b in beta:
      itList.append(b)
   iterTable = [itList]

   for it in range(maxIter):
      gradient = numpy.dot(designXT, (y - y_predProb))
      dispersion = y_predProb * (1.0 - y_predProb)
      hessian = - numpy.dot(designXT, (dispersion.reshape((nObs,1)) * designX))
      delta = numpy.linalg.solve(hessian, gradient)
      step = 1.0
      for iStep in range(maxStep):
         beta_next = beta - step * delta
         nu_next = numpy.dot(designX, beta_next)
         odds = numpy.exp(nu_next)
         y_p0 = 1.0 / (1.0 + odds) 
         llk_next = numpy.sum(y * numpy.log(odds) + numpy.log(y_p0))
         if ((llk_next - llk) > - tolLLK):
            break
         else:
            step = 0.5 * step

      diffBeta = beta_next - beta
      llk = llk_next
      beta = beta_next
      y_predProb = 1.0 - y_p0
      itList = [it+1, llk]
      for b in beta:
         itList.append(b)
      iterTable.append(itList)
      if (numpy.linalg.norm(diffBeta) < tolBeta):
         break

   dispersion = y_predProb * (1.0 - y_predProb)
   hessian = - numpy.dot(designXT, (dispersion.reshape((nObs,1)) * designX))
   covBeta = numpy.linalg.inv(hessian)

   return(iterTable, nonAliasParam, beta, covBeta, y_predProb)

inputData = pandas.read_csv('C:\\MScAnalytics\\Linear and Nonlinear Model\\Data\\hmeq.csv')

trainData = inputData[['BAD','REASON','JOB']].dropna()
y = trainData['BAD']

columnJOB = pandas.get_dummies(trainData[['JOB']].astype('category'))
columnREASON = pandas.get_dummies(trainData[['REASON']].astype('category'))
columnJOB_REASON = create_interaction (columnJOB, columnREASON)

# Intercept only model
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(None, y)

# Intercept + JOB model
X = columnJOB
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + JOB + JOB * REASON model
X = columnJOB.join(columnJOB_REASON)
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + REASON model
X = columnREASON
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + REASON + JOB * REASON model
X = columnREASON.join(columnJOB_REASON)
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + JOB + REASON model
X = columnJOB.join(columnREASON)
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + JOB + REASON + JOB * REASON model
X = columnJOB.join(columnREASON)
X = X.join(columnJOB_REASON)
iterTable, nonAliasParam, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)