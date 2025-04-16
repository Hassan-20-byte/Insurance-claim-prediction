# Insurance-claim-prediction
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_data = pd.read_csv('C:\\Users\\user\\Desktop\\IP data\\train_data.csv')
train_data.head()
train_data.info()
train_data.columns
var_desc = pd.read_csv('C:\\Users\\user\\Desktop\\IP data\\VariableDescription.csv')
Sub_sam = pd.read_csv('C:\\Users\\user\\Desktop\\IP data\\SampleSubmission (1).csv')
test_data = pd.read_csv('C:\\Users\\user\\Desktop\\IP data\\test_data.csv')
test_data.head()
test_data.isnull().sum()
train_data.shape
test_data_shape
train_data['Garden'].value_counts()
train_data['YearOfObservation'].value_counts()
sns.catplot(x='Claim', kind = 'count', data = train_data)
X_train = train_data.drop(['Claim'], axis = 1)
Y_train = train_data['Claim']
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values =np.nan, strategy = 'most_frequent')
X_train = X_train.drop(['Customer Id'], axis = 1)
X_train['Building_Type'].nunique()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

One_hot = OneHotEncoder()
le_en = LabelEncoder()
X_train['Geo_Code'].nunique()
(X_train['Date_of_Occupancy'])
_train['Geo_Code'] = X_train['Geo_Code'].astype(object)
X_train.info()
X_train['NumberOfWindows'].values.astype(float)
from sklearn.preprocessing import MaxAbsScaler
def preprocessing(data):
    float_array = data[['Building_Type', 'YearOfObservation','Residential',]].values.astype(float)
    ##Label Encoder Conversion
    
    data['Garden'] = le_en.fit_transform(data['Garden'])
    data['Settlement'] =le_en.fit_transform(data['Settlement'])
    data['Residential'] = le_en.fit_transform(data['Residential'])
    data['Building_Painted'] = le_en.fit_transform(data['Building_Painted'])
    data['Building_Fenced']  = le_en.fit_transform(data['Building_Fenced'])
    ##drop customer id
    data = data.drop(['Customer Id'], axis = 1)
    
    categ = ['Geo_Code', 'NumberOfWindows']
    data = pd.get_dummies(data, prefix_sep="_", columns=categ)
    
 
    

    
    scaler = MaxAbsScaler()
    data = scaler.fit_transform(data)
    missing   = data[['Geo_Code','Date_of_Occupancy','NumberOfWindows']]
    data = impute.fit_transform(data)
    
    
    return data

    processed__train = preprocessing(X_train)
    processed_test = preprocessing(test_data)
    X_train[['Geo_Code','Date_of_Occupancy','NumberOfWindows']]
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(processed__train, Y_train, stratify =Y_train, test_size = 0.2, random_state = 45)
    rom sklearn.linear_model import LogisticRegression
##using logistic regression

lg =LogisticRegression(max_iter = 1000)
lg.fit(X_train,Y_train)
from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier()

knn.fit(X_train,Y_train)
    
print('knn classifier :', 1 -accuracy_score(Y_test, kn_model))
rom sklearn.metrics import accuracy_score
lg_model = lg.predict(X_test)
print('Error rate of Logistic Classifier', 1 -accuracy_score(Y_test, lg_model))
##tunning logistic regression paramters
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
lg_r =LogisticRegression(solver = 'saga', max_iter = 5000)
weights = np.linspace(0.0, 0.99, 10)
params ={'C':[0.0001,0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'class_weight':[{0:x, 1:1.0 - x} for x in weights]}

model = GridSearchCV(estimator = lg_r, param_grid = params,
                     scoring =accuracy_score, cv = 5, verbose = 1, n_jobs = 1)

    
    
    
