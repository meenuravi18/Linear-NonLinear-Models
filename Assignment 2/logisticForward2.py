import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as sstats
pandas.set_option('precision', 6)
#pandas.set_option('display.float_format', '{:.6E}'.format)
pandas.set_option('display.max_columns', 10)
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

inputData = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/claim_history.csv', delimiter = ',')

trainData = inputData[inputData['CLM_COUNT'] > 0.0].dropna()

y = trainData['CLM_AMT'] / trainData['CLM_COUNT']

candidate_name =['HOMEKIDS', 'KIDSDRIV', 'CAR_TYPE', 'CAR_AGE']
candidate_measure = ['interval', 'interval', 'categorical', 'interval']

# trainData = inputData[['claim_indicator'] + catlist].dropna()
# y = trainData['claim_indicator']

# columnHK = pandas.get_dummies(trainData[['HOMEKIDS']].astype('interval'))
# columnKD = pandas.get_dummies(trainData[['KIDSDRIV']].astype('interval'))
# columnCT = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category'))
# columnCA = pandas.get_dummies(trainData[['CAR_AGE']].astype('interval'))

for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      X = pandas.get_dummies(trainData[[X_name]].astype('category'))
   else:
      X = trainData[[X_name]]


table=pandas.DataFrame(index=[0,1,2,3,4])
table['parameters']=['Intercept','HOMEKIDS', 'KIDSDRIV', 'CAR_TYPE', 'CAR_AGE']
freeParam=[]
llh=[]
deviance=[]
dof=[]
css=[]
# Step 0# Intercept only
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(None, y)
freeParam.append(len(nonAliasParam))
llh.append(llk)

# # Step 1
# # Intercept + hk model
X = columnHK
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
freeParam.append(len(nonAliasParam))
llh.append(llk)


# Intercept + kd
X = columnM
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
freeParam.append(len(nonAliasParam))
llh.append(llk)

# Intercept + ct
X = columnPAT
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
freeParam.append(len(nonAliasParam))
llh.append(llk)

# Intercept + ca
X = columnPG
iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
freeParam.append(len(nonAliasParam))
llh.append(llk)



for i in range(0,len(llh)):
  if i==0:
    deviance.append("-")
  else:
    deviance.append((llh[i]-llh[0])*2)
for i in range(0,len(freeParam)):
  if i==0:
    dof.append("-")
  else:
    dof.append((freeParam[i]-freeParam[0]))

for i in range(0,len(deviance)):
  if i==0:
    css.append("-")
  else:
    temp= sstats.chi2.sf(deviance[i],dof[i])
    temp=format(temp, '.4E')
    css.append(temp)

table['freeParam']=freeParam
table['log liklihood']=llh
table['deviance']=deviance
table['degree of freedom']=dof
table['Chi square significance']=css
#table.to_csv('table.csv',index=True,header=True)
print(table)


# table=pandas.DataFrame(index=[0,1,2,3])
# table['parameters']=['Intercept+URBANICITY','MSTATUS', 'CAR_TYPE', 'REVOKED']
# freeParam=[freeParam[-1]]
# llh=[llh[-1]]
# deviance=[]
# dof=[]
# css=[]
# # # Step 2
# X0 = columnPG

# # Intercept+URBANICITY + mstatus
# X = X0.join(columnAT)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
# freeParam.append(len(nonAliasParam))
# llh.append(llk)

# # Intercept+URBANICITY + car type
# X = X0.join(columnM)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
# freeParam.append(len(nonAliasParam))
# llh.append(llk)

# # Intercept+URBANICITY + revoked
# X = X0.join(columnPAT)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
# freeParam.append(len(nonAliasParam))
# llh.append(llk)


# for i in range(0,len(llh)):
#   if i==0:
#     deviance.append("-")
#   else:
#     deviance.append((llh[i]-llh[0])*2)
# for i in range(0,len(freeParam)):
#   if i==0:
#     dof.append("-")
#   else:
#     dof.append((freeParam[i]-freeParam[0]))

# for i in range(0,len(deviance)):
#   if i==0:
#     css.append("-")
#   else:
#     temp= sstats.chi2.sf(deviance[i],dof[i])
#     temp=format(temp, '.4E')
#     css.append(temp)

# table['freeParam']=freeParam
# table['log liklihood']=llh
# table['deviance']=deviance
# table['degree of freedom']=dof
# table['Chi square significance']=css
# #table.to_csv('table.csv',index=True,header=True)
# #print(table)

# # Step 3
# table=pandas.DataFrame(index=[0,1,2])
# table['parameters']=['Intercept+URBANICITY+CAR_TYPE','MSTATUS', 'REVOKED']
# freeParam=[freeParam[-2]]
# llh=[llh[-2]]
# deviance=[]
# dof=[]
# css=[]
# X0 = X0.join(columnM)

# # Intercept+URBANICITY+CAR_TYPE + mstatus
# X = X0.join(columnAT)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
# freeParam.append(len(nonAliasParam))
# llh.append(llk)

# # Intercept+URBANICITY+CAR_TYPE + revoked
# X = X0.join(columnPAT)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
# freeParam.append(len(nonAliasParam))
# llh.append(llk)


# for i in range(0,len(llh)):
#   if i==0:
#     deviance.append("-")
#   else:
#     deviance.append((llh[i]-llh[0])*2)
# for i in range(0,len(freeParam)):
#   if i==0:
#     dof.append("-")
#   else:
#     dof.append((freeParam[i]-freeParam[0]))

# for i in range(0,len(deviance)):
#   if i==0:
#     css.append("-")
#   else:
#     temp= sstats.chi2.sf(deviance[i],dof[i])
#     temp=format(temp, '.4E')
#     css.append(temp)

# table['freeParam']=freeParam
# table['log liklihood']=llh
# table['deviance']=deviance
# table['degree of freedom']=dof
# table['Chi square significance']=css
# # table.to_csv('table.csv',index=True,header=True)
# # print(table)



# # Step 4
# table=pandas.DataFrame(index=[0,1])
# table['parameters']=['Intercept+URBANICITY+CAR_TYPE+MSTATUS', 'REVOKED']
# freeParam=[freeParam[1]]
# llh=[llh[1]]
# deviance=[]
# dof=[]
# css=[]
# X0 = X0.join(columnAT)

# # Intercept+URBANICITY+CAR_TYPE+MSTATUS + revoked
# X = X0.join(columnPAT)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)
# freeParam.append(len(nonAliasParam))
# llh.append(llk)

# for i in range(0,len(llh)):
#   if i==0:
#     deviance.append("-")
#   else:
#     deviance.append((llh[i]-llh[0])*2)
# for i in range(0,len(freeParam)):
#   if i==0:
#     dof.append("-")
#   else:
#     dof.append((freeParam[i]-freeParam[0]))

# for i in range(0,len(deviance)):
#   if i==0:
#     css.append("-")
#   else:
#     temp= sstats.chi2.sf(deviance[i],dof[i])
#     temp=format(temp, '.4E')
#     css.append(temp)

# table['freeParam']=freeParam
# table['log liklihood']=llh
# table['deviance']=deviance
# table['degree of freedom']=dof
# table['Chi square significance']=css
# table.to_csv('table.csv',index=True,header=True)
# print(table)


# # Step 5
# # + primary_age_tier * primary_gender
# columnPAT_PG = create_interaction (columnPAT, columnPG)
# X = X0.join(columnPAT_PG)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# # + primary_age_tier * residence_location
# columnPAT_RL = create_interaction (columnPAT, columnRL)
# X = X0.join(columnPAT_RL)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# # + primary_gender * residence_location
# columnPG_RL = create_interaction (columnPG, columnRL)
# X = X0.join(columnPG_RL)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X, y)

# # Final main effect + interaction model
# X0 = X0.join(columnPG_RL)
# iterTable, nonAliasParam, llk, beta, covBeta, y_predProb = GLM4BinaryLogistic(X0, y)

# paramName = X0.columns.insert(0,'Intercept')
# beta = pandas.Series(beta, index = paramName[nonAliasParam])
