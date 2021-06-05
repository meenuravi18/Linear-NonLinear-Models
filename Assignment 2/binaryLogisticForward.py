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

   return(iterTable, nonAliasParam, llk, beta, covBeta, y_predProb)

HO_claim_history = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/HO_claim_history.csv', delimiter = ',')

HO_claim_history[['claim_indicator']] = numpy.where(HO_claim_history[['num_claims']] > 0, 1, 0)

catlist = ['aoi_tier','marital','primary_age_tier','primary_gender','residence_location']

trainData = HO_claim_history[['claim_indicator'] + catlist].dropna()
y = trainData['claim_indicator']
y.to_csv("bab.csv",index=True)
columnAT = pandas.get_dummies(trainData[['aoi_tier']].astype('category'))
columnM = pandas.get_dummies(trainData[['marital']].astype('category'))
columnPAT = pandas.get_dummies(trainData[['primary_age_tier']].astype('category'))
columnPG = pandas.get_dummies(trainData[['primary_gender']].astype('category'))
columnRL = pandas.get_dummies(trainData[['residence_location']].astype('category'))

# Step 0
# Intercept only
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(None, y)

# Step 1
# Intercept + aoi_tier model
X = columnAT
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + marital model
X = columnM
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + primary_age_tier
X = columnPAT
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + primary_gender
X = columnPG
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location
X = columnRL
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Step 2
X0 = columnRL

# Intercept + residence_location + aoi_tier model
X = X0.join(columnAT)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location + marital model
X = X0.join(columnM)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location + primary_age_tier
X = X0.join(columnPAT)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location + primary_gender
X = X0.join(columnPG)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Step 3
X0 = X0.join(columnPAT)

# Intercept + residence_location + primary_age_tier + aoi_tier
X = X0.join(columnAT)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location + primary_age_tier + marital
X = X0.join(columnM)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location + primary_age_tier + primary_gender
X = X0.join(columnPG)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Step 4
X0 = X0.join(columnPG)

# Intercept + residence_location + primary_age_tier + primary_gender + aoi_tier
X = X0.join(columnAT)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Intercept + residence_location + primary_age_tier + primary_gender + marital
X = X0.join(columnM)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Final main effect model
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X0, y)

paramName = X0.columns.insert(0,'Intercept')
beta = pandas.Series(beta, index = paramName[nonAliasParam])

# Step 5
# + primary_age_tier * primary_gender
columnPAT_PG = create_interaction (columnPAT, columnPG)
X = X0.join(columnPAT_PG)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# + primary_age_tier * residence_location
columnPAT_RL = create_interaction (columnPAT, columnRL)
X = X0.join(columnPAT_RL)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# + primary_gender * residence_location
columnPG_RL = create_interaction (columnPG, columnRL)
X = X0.join(columnPG_RL)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# Final main effect + interaction model
X0 = X0.join(columnPG_RL)
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X0, y)

paramName = X0.columns.insert(0,'Intercept')
beta = pandas.Series(beta, index = paramName[nonAliasParam])
print(beta)
