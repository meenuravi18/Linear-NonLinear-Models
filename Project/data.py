import pandas as pd 
import matplotlib.pyplot as plt

inputData = pd.read_csv('C:/Users/meenu/OneDrive/LinearNonLinearModels/Project/fleet_truck.csv',
                            delimiter = ',')
mf_0=0
mf_1=0
mf=0
counts=inputData['Maintenance_flag'].value_counts().to_dict()
percents=[]
# percentZero=mf_0/mf
# percentOne=mf_1/mf
percents.append(counts[0]/(counts[0]+counts[1])*100)
percents.append(counts[1]/(counts[0]+counts[1])*100)
labs=['No Maintenance_flag','Maintenance_flag']
mycolors=['darkslateblue','lightsteelblue']
plt.pie(percents,labels=labs,colors = mycolors)
plt.show()
# df = pd.read_csv('data1.csv')