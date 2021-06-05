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


inputData = pandas.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Assignment 2/claim_history.csv',
                            delimiter = ',',usecols=['MSTATUS', 'CAR_TYPE', 'REVOKED', 'URBANICITY','CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME','CLM_COUNT','EXPOSURE'])

inputData=inputData.dropna().reset_index(drop=True)
catlist = ['MSTATUS', 'CAR_TYPE', 'REVOKED', 'URBANICITY','CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME','CLM_COUNT','EXPOSURE']
inputData['freq']=inputData['CLM_COUNT']/inputData['EXPOSURE']
inputData[['target']] = numpy.where(inputData[['freq']] > 1, 1, 0)
trainData = inputData[['target'] + catlist].dropna().reset_index(drop=True)

trainData_ = inputData[['target','freq','EXPOSURE'] + catlist].dropna()
y_train = trainData['target']


dframe=pandas.DataFrame(index=[0,1,2,3,4,5,6,7,8])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Intercept only model
iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (None, y_train)
# outTable = pandas.DataFrame(iterTable, columns = ['Iteration', 'Log-Likelihood', 'Scale', 'Intercept'])
# print('Iteration History Table:\n', outTable)
# print('\n')
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
# Enter the first predictor
candidate_name = ['MSTATUS', 'CAR_TYPE', 'REVOKED', 'URBANICITY','CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME']
candidate_measure = ['categorical', 'categorical', 'categorical', 'categorical','interval','interval','interval','interval']

for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      X_train = pandas.get_dummies(trainData[[X_name]].astype('category'))
   else:
      X_train = trainData[[X_name]]
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)

   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]
     
   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print("Log-Likelihood",llk)
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
dframe.to_csv("out.csv",index=True,header=True)

print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1,2,3,4,5,6,7])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# # Enter the second predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['MSTATUS', 'CAR_TYPE', 'REVOKED','CAR_AGE', 'MVR_PTS', 'TIF', 'TRAVTIME']
candidate_measure = ['categorical', 'categorical','categorical','interval','interval','interval','interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]
   
   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', df)
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
#dframe.to_csv("out.csv",index=True,header=True)




print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1,2,3,4,5,6])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Enter the third predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = X0_train.join(trainData[['MVR_PTS']])
iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['MSTATUS', 'CAR_TYPE', 'REVOKED','CAR_AGE', 'TIF', 'TRAVTIME']
candidate_measure = ['categorical', 'categorical','categorical','interval','interval','interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]


   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
#dframe.to_csv("out.csv",index=True,header=True)


print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1,2,3,4,5])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Enter the fourth predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = X0_train.join(trainData[['MVR_PTS', 'CAR_AGE']])

iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['MSTATUS', 'CAR_TYPE', 'REVOKED', 'TIF', 'TRAVTIME']
candidate_measure = ['categorical', 'categorical','categorical','interval','interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]


   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
#dframe.to_csv("out.csv",index=True,header=True)
print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1,2,3,4])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Enter the fifth predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = pandas.get_dummies(trainData[['MSTATUS']].astype('category')).join(X0_train)
X0_train = X0_train.join(trainData[['MVR_PTS', 'CAR_AGE']])

iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['CAR_TYPE', 'REVOKED', 'TIF', 'TRAVTIME']
candidate_measure = ['categorical','categorical','interval','interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]


   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
#dframe.to_csv("out.csv",index=True,header=True)
print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1,2,3])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Enter the sixth predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = pandas.get_dummies(trainData[['MSTATUS']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['REVOKED']].astype('category')).join(X0_train)
X0_train = X0_train.join(trainData[['MVR_PTS', 'CAR_AGE']])

iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['CAR_TYPE', 'TIF', 'TRAVTIME']
candidate_measure = ['categorical','interval','interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]


   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
#dframe.to_csv("out.csv",index=True,header=True)
print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1,2])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Enter the seventh predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = pandas.get_dummies(trainData[['MSTATUS']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['REVOKED']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category')).join(X0_train)
X0_train = X0_train.join(trainData[['MVR_PTS', 'CAR_AGE']])

iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['TIF', 'TRAVTIME']
candidate_measure = ['interval','interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]


   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
#dframe.to_csv("out.csv",index=True,header=True)
print("*******************************************************************************************************************************")
dframe=pandas.DataFrame(index=[0,1])
model=[]
numParam=[]
llh=[]
dev=[]
dof=[]
css=[]
# Enter the eigth predictor
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = pandas.get_dummies(trainData[['MSTATUS']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['REVOKED']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category')).join(X0_train)
X0_train = X0_train.join(trainData[['MVR_PTS', 'CAR_AGE','TRAVTIME']])

iterTable, nonAliasParam, llk0, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
df0 = len(nonAliasParam)
model.append('intercept')
numParam.append(df0)
llh.append(llk0)
dev.append("-")
dof.append("-")
css.append("-")
print('Intercept non alias parameter',df0)
print('intercept llk',llk0)
candidate_name = ['TIF']
candidate_measure = ['interval']
for X_name, X_measure in zip(candidate_name, candidate_measure):
   if (X_measure == 'categorical'):
      col = pandas.get_dummies(trainData[[X_name]].astype('category'))
      X_train = X0_train.join(col)
   else:
      X_train = X0_train.join(trainData[[X_name]])
   iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X_train, y_train)
   df = len(nonAliasParam)
   devianceChiSq = 2.0 * (llk - llk0)
   devianceDF = df - df0
   deviancePValue = sstats.chi2.sf(devianceChiSq, devianceDF)
   X_column = ['Intercept'] + X_train.columns.to_list()
   param_name = [X_column[i] for i in nonAliasParam]


   print('Enter Predictors: ', X_name)
   print('Parameters: ', param_name)
   print('Number of Non-Aliased Parameters = ', len(param_name))
   print('Chi-Square =', devianceChiSq)
   print('        DF =', devianceDF)
   print('   P-Value =', deviancePValue)
   model.append(X_name)
   numParam.append(len(param_name))
   llh.append(llk)
   dev.append(devianceChiSq)
   dof.append(devianceDF)
   css.append(deviancePValue)
dframe['parameters']=model   
dframe['num Non-Aliased']=numParam
dframe['llh']=llh
dframe['dev']=dev
dframe['dof']=dof
dframe['css']=css
# dframe.to_csv("out.csv",index=True,header=True)
# # Final model
X0_train = pandas.get_dummies(trainData[['URBANICITY']].astype('category'))
X0_train = pandas.get_dummies(trainData[['MSTATUS']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['REVOKED']].astype('category')).join(X0_train)
X0_train = pandas.get_dummies(trainData[['CAR_TYPE']].astype('category')).join(X0_train)
X0_train = X0_train.join(trainData[['MVR_PTS', 'CAR_AGE','TRAVTIME','TIF']])


iterTable, nonAliasParam, llk, beta, covBeta, y_pred = GLM4BinaryLogistic (X0_train, y_train)
X_column = ['Intercept'] + X0_train.columns.to_list()
param_name = [X_column[i] for i in nonAliasParam]

# # Final parameter estimates
outCoefficient = pandas.Series(beta, index = param_name)
print('Parameter Estimates:\n', outCoefficient, numpy.exp(outCoefficient))


#Question 3a
plt.figure(figsize = (7,4), dpi = 200)
plt.scatter(trainData_['freq'],y_pred, marker = 'o', c = trainData_['EXPOSURE'], s = 20)
plt.colorbar().set_label('EXPOSURE')
plt.xlabel('Observed Frequency')
plt.ylabel(' predicted Event probability')
#plt.xticks(numpy.arange(0,1,0.2))
#plt.yticks(numpy.arange(0,5000,500))
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()
#Question 3b
frequencies=trainData_['target'].to_list()
yi=numpy.array(frequencies)
pi=numpy.array(y_pred)
resDev=numpy.where(yi==0,-1*numpy.sqrt(2*numpy.log(1/(1-pi))),numpy.sqrt(2*yi*numpy.log(yi/pi)))

plt.figure(figsize = (7,4), dpi = 200)
plt.scatter(trainData_['freq'],resDev, marker = 'o', c = trainData['EXPOSURE'], s = 20)
plt.colorbar().set_label('EXPOSURE')
plt.xlabel('Observed Frequency')
plt.ylabel('Deviance Residual')
#plt.xticks(numpy.arange(0,1,0.2))
#plt.yticks(numpy.arange(0,5000,500))
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()

#Question 4
eventPred=numpy.where(y_pred>=0.25,1,0)

accuracy=pandas.DataFrame(columns=['y_pred','event_predictor'])
accuracy['y_pred']=y_pred
accuracy['event_predictor']=eventPred
accuracy['event_obs']=y_train
accuracy['actual']=numpy.where(accuracy['event_predictor']==accuracy['event_obs'],1,0)
accuracyMetric=accuracy['actual'].value_counts()
print((accuracy['actual'] == 1).sum()/len(accuracy))
accuracy.to_csv("a.csv",index=True)
