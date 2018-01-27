import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import pandas as pd
import numpy as np

#the main.csv is the file that contains all the tables together!
kbc_bank_info 	= pd.read_csv('/Users/Fernando/Documents/Jobs/KBC/data_science/main.csv')

cust_req 		= pd.DataFrame({
	'Category': {0: 'Credit', 1: 'Loan',2:'Mutual Fund'}, 
	'Sum': {
	0: float(kbc_bank_info[["Sale_CC"]].sum()),
	1:float(kbc_bank_info[["Sale_CL"]].sum()),
	2:float(kbc_bank_info[["Sale_MF"]].sum()) }})

print(kbc_bank_info[["Sale_CC"]].sum())
print(kbc_bank_info[["Sale_CL"]].sum())
print(kbc_bank_info[["Sale_MF"]].sum())


sns.barplot(x = 'Sum', y = 'Category', data = cust_req)
plt.title('Requirement of Customers')
plt.xlabel('Count')
plt.ylabel('Marketing Scheme')

kbc_bank_info = kbc_bank_info.dropna()
corr = kbc_bank_info.corr()
corr.sort_values(["Sale_CL"], ascending = False, inplace = True)
print(corr.Sale_CL)

data_loan = kbc_bank_info[['Client', 'Sale_CL','Revenue_CL','Tenure','ActBal_MF','Count_CL', 'ActBal_OVD', 'ActBal_CL', 'Count_SA', 'TransactionsCred_CA', 'Count_CA', 'TransactionsCred', 'TransactionsDebCash_Card']]

X = data_loan.ix[:,'Tenure':'Client'].as_matrix()
y = data_loan.Sale_CL.as_matrix()

Corr_Loan = data_loan.corr()
plt.figure(figsize=(10,10))
sns.heatmap(Corr_Loan, vmax=1, square=True,annot=True,cmap='seismic')

plt.title('Correlation between the selected Features')