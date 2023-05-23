# import
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import seaborn as sns  


#data load
ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\stock prodiction (svr)11project ml\NSE-Tata-Global-Beverages-Limited.csv")
#summary
print(ds.head(4))
print(ds.tail(4))
print(ds.shape)
print(ds.info())
print(ds.describe())
#graph
"""sns.pairplot(ds)
plt.show()"""
#drop"""
ds=ds.drop("Date",axis="columns")
ds=ds.drop("Total Trade Quantity",axis="columns")
 
#x and y finding

x=ds.iloc[:,:-1].values
y=ds[['Close']].values
y=y.reshape(len(y),1)
print(x)
print(y)
#train_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)

#feuture scaling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#model train
model=LinearSVR()
model.fit(x_train,y_train)
#prediction
pred=model.predict(x_test)

#using scatter plot compare the actual and predicted data
plt.figure(figsize=(12,6))
plt.scatter(y_test,pred)
plt.title('Actual versus Prediction ')
plt.xlabel('Actual', fontsize=20)                         
plt.ylabel('Predicted', fontsize=20)
plt.show()
#output

print(r2_score(y_test,pred))
print(mean_squared_error(y_test,pred))