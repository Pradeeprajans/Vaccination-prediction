# import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score,classification_report,confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# importing the dataset


df = pd.merge(pd.read_csv("features.csv"),pd.read_csv("labels.csv"),right_index=True,left_index=True)

print(df)

df.drop("respondent_id_y",axis=1,inplace=True)

df.rename({"respondent_id_x":"id"},axis=1,inplace=True)

# Define our X and y
X = df.drop(columns = ['id', 'h1n1_vaccine', 'seasonal_vaccine'], axis=1)
y = df['h1n1_vaccine']


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Set up lists for each columns datatypes
num_cols = []
ohe_cols = []
freq_cols = []

for c in X.columns:
    if X[c].dtype in ['float64', 'int64']:
        num_cols.append(c)
    elif X[c].nunique() < 10:
        ohe_cols.append(c)
    else:
        freq_cols.append(c)
        

# We wanted to see each column category
print(f'Numerical Columns:', num_cols)
print('\n')
print(f'Object Columns (with less than 10 unique values):', ohe_cols)
print('\n')
print(f'Object Columns (with more than 10 unique values):', freq_cols)


# Handle numeric columns

numeric_imputer = IterativeImputer(max_iter=100, random_state=42)
x_train_numeric = numeric_imputer.fit_transform(x_train[num_cols])
x_test_numeric = numeric_imputer.transform(x_test[num_cols])
scaler = MinMaxScaler()
x_train_numeric = scaler.fit_transform(x_train_numeric)
x_test_numeric = scaler.transform(x_test_numeric)

# Handle categorical columns

categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
x_train_categorical = categorical_imputer.fit_transform(x_train[ohe_cols])
x_test_categorical = categorical_imputer.transform(x_test[ohe_cols])
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
x_train_categorical = encoder.fit_transform(x_train_categorical)
x_test_categorical = encoder.transform(x_test_categorical)

# Convert results back to DataFrame
x_train_numeric = pd.DataFrame(x_train_numeric, columns=num_cols)
x_test_numeric = pd.DataFrame(x_test_numeric, columns=num_cols)
x_train_categorical = pd.DataFrame(x_train_categorical, columns=encoder.get_feature_names_out(ohe_cols))
x_test_categorical = pd.DataFrame(x_test_categorical, columns=encoder.get_feature_names_out(ohe_cols))

# Concatenate everything back into a single DataFrame
x_train_processed = pd.concat([x_train_numeric, x_train_categorical], axis=1)
x_test_processed = pd.concat([x_test_numeric, x_test_categorical], axis=1)


smote = SMOTE(random_state=42) 
x_train_smote, y_train_smote = smote.fit_resample(x_train_processed, y_train) 

# Rename columns to valid string names
x_train_processed.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in x_train_processed.columns]
x_test_processed.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace(' ', '_') for col in x_test_processed.columns]



# Create a dictionary to store the fitted preprocessors
preprocessors = {
    'numeric_imputer': numeric_imputer,
    'scaler': scaler,
    'categorical_imputer': categorical_imputer,
    'encoder': encoder
}

# Save the preprocessors to a pickle file
with open('preprocessors.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

print("Preprocessors saved to preprocessors.pkl")


# Assuming GBHT is your trained Gradient Boosting Classifier model
final_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, max_depth=9, random_state=28)
final_model.fit(x_train_smote, y_train_smote)

# Save the model to a pickle file
with open('h1n1_gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Model saved to gradient_boosting_model.pkl")
