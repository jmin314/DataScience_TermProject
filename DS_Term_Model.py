import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from matplotlib.pyplot import plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocessing

# Read base Data set
df = pd.read_csv('/Users/jangminseong/Desktop/데과/텀프/LA_Crime_Data_2022.csv')
df2023 = pd.read_csv('/Users/jangminseong/Desktop/데과/텀프/Crime_2023.csv')

# User Input Data example
User_input_data = {'TIME OCC' : [21],
                   'AREA' : [15],
                   'Vict Age' : [47],
                   'Vict Sex' : ['F'],
                   'Vict Descent' : ['B']
                   }
User_Input = pd.DataFrame(User_input_data)

# Preporcessing User input data
UserInputPreprocessed = Preprocessing.User_Input_Preprocessing(User_Input)

# Preprocessing base Data set
Preprocessed_2022 = Preprocessing.For_Full_Data_Model(df)
# Preprocessed_2023 = Preprocessing.For_Full_Data_Model(df2023)
# Standard Scaling base Data set
Scalled_2022 = Preprocessing.Scaling_Set(Preprocessed_2022)
# Scalled_2023 = Preprocessing.Scaling_Set(Preprocessed_2023)

# Set X, y for Full Data Model
X = Scalled_2022.drop(['Felony Rate By TA', 'Freq F By Hour', 'Vict Age', 'Freq F By Age'], axis = 1)
y = Scalled_2022['Felony Rate By TA']
# X_2023 = Scalled_2023.drop(['Felony Rate By TA', 'Freq F By Hour', 'Vict Age', 'Freq F By Age'], axis = 1)

# Set X, y for User Input Data Model
drop_columns=['TIME OCC', 'AREA', 'Vict Age', 'Felony Rate By TA','Weapon Or Not', 'Crime Class', 'Crime Count',
              'Average CC By TA', 'Average CClass By TA', 'Ex Convict', 'Average Weapon By TA', 'Freq F By Status', 'Average Ex Convict By TA']
XU = Preprocessed_2022.drop(columns=drop_columns, axis = 1)
yU = Preprocessed_2022['Crime Class']

# Merge yU's data into 0 and 1(1 merge 2)
filtered_indices = yU[(yU == 1) | (yU == 2)].index
yU.loc[filtered_indices] = 1

# Split data for Full data model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

XU_train, XU_test, yU_train, yU_test = train_test_split(XU, yU, test_size=0.2, random_state=42)

# Fit train data
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Claculate MSE using K Fold
k = 10  
kf = KFold(n_splits=k, shuffle=True)
mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
mean_mse = np.mean(mse_scores)
print(f'Mean Squared Error: {mean_mse}')

# Fit all XU and yU
modelU = LogisticRegression()
modelU.fit(XU_train, yU_train)

# Prediction
y_pred = model.predict(X_test)
print("Full Dataset Test:")
num_results = 10  
for i in range(num_results):
    print(f"Example {i+1}: {y_pred[i]:.6f}")

# Model score
print("Score", model.score(X_test, y_test))
    
# y_2023_pred = model.predict(X_2023)
# yU_pred = modelU.predict_proba(XU_test)
yU_pred = modelU.predict_proba(UserInputPreprocessed)


    
# print("2023 Dataset Test:")
# num_results = 10  
# for i in range(num_results):
#     print(f"Example {i+1}: {y_2023_pred[i]:.6f}")

# Result about User Input Data
print("User Input prediction : \n", yU_pred)
print("User Input Model Accuracy : ", modelU.score(XU_test, yU_test))

# Each X's correlation matrix
correlation_matrixX = X.corr()
# Correlatioin matrix between X and y
correlation_matrixXY = pd.concat([X, y], axis=1).corr()

# Drawing Heatmap according to correlation matrix X
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrixX, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap of X')
plt.show()

# Drawing Scatter plot according to correlation matrix XY
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(range(len(correlation_matrixXY.columns)), correlation_matrixXY['Felony Rate By TA'], c='blue', label='Correlation')
ax.set_xticks(range(len(correlation_matrixXY.columns)))
ax.set_xticklabels(correlation_matrixXY.columns, rotation=45, ha='right')
ax.set_ylabel('Correlation with Felony Rate')
ax.set_title('Correlation Scatter Plot')
ax.legend()
plt.show()

# Calculate MSE and RMSE about Full Data Model
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)
