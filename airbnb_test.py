# Importing usual libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Importing ml libraries
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler


############################## INITIAL DATA READ, SETUP, AND PREPROCESSING #######################################
# Reads in csv file, encoding argument avoids unicode decode error
df_full = pd.read_csv('https://raw.githubusercontent.com/cortmcelmury/air_bnb/master/data/Portland_AirBnb_Listings_Test.csv',
                      encoding='ISO-8859-1', index_col='id')

# Look at structure of dataframe (num columns, types, and nulls)
df_full.info()

df_full.head()

# subset df to variables I want to use
df = df_full[['review_scores_rating', 'host_since', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'host_listings_count',
              'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'price', 'security_deposit',
              'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights']]

df.info()
df.head()

############################## Begin Preprocessing #############################
# Convert categorical columns to type category
cat_cols = ['room_type', 'bed_type']

# The warning that shows up is ok
df[cat_cols] = df[cat_cols].apply(
    lambda x: x.astype('category'), axis=0)

# Remove rows where target is null
df.review_scores_rating.isnull().sum()
df = df.dropna(subset=['review_scores_rating'])

df.info()

# Convert remaining currency variables to float
currency_cols = ['price', 'security_deposit', 'cleaning_fee', 'extra_people']

df[currency_cols] = df[currency_cols].replace(
    '[\$,]', '', regex=True).astype(float)

# Deal with host_since variable (date)
# Convert to a datetime variable
df['host_since'] = pd.to_datetime(df['host_since'])

# Create variable that lists how long a host has been a host
df['years_as_host'] = (dt.datetime.now().year - df['host_since'].dt.year)

# Drop host_since variable, won't be used for modeling
df = df.drop(['host_since'], axis=1)

# Fill nulls
# For security_deposit and cleaning_fee, fill with 0 since we can assume blank means 0
# Drop the nulls that are left (small number of records)
df.isnull().sum()

df[['security_deposit', 'cleaning_fee']] = df[[
    'security_deposit', 'cleaning_fee']].fillna(0)

# Drop remaining nulls - only 5 records
df = df.dropna()

df.head()

# Convert t/f to 1 and 0
tf_dict = {'t': 1,
           'f': 0}

tf_cols = ['host_is_superhost',
           'host_has_profile_pic', 'host_identity_verified']
df[tf_cols] = df[tf_cols].replace(tf_dict)

# Look at each categorical feature and its values
for col in cat_cols:
    print(df[col].value_counts())
    print('\n')


######### Get dummies - one-hot encoding for categorical variables #############
# Create a new df that has the cat_cols split out into binary variables
# drop_first arg specifies whether or not to keep the first variable
# Example: for room_type, there is 'entire home/apt', 'private room', and 'shared room'
# Once split to binary vars, if 'private room' and 'shared room' are 0, we know 'entire home/apt' must be 1
# If interpretability is key, drop_first should be set to false
df_coded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
df_coded.head()


# Get features into X for modeling (drop the target variable)
X = df_coded.drop(['review_scores_rating'], axis=1)

# Get only review_scores_rating (target variable)
y = df_coded.values[:, 0]

# Create train and test sets (split 75/25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=23, shuffle=True)

# Convert arrays back to type int for evaluation purposes
y_train = y_train.astype('int')
y_test = y_test.astype('int')

X_train.columns

X_train.info()

######## Scaling not needed since XGBoost scales numerics inherently ###########

################################## NAIVE TEST TO SEE IF MODEL IMPROVES ################################################
# Simply predict the mean of the review scores for every single observation and calculate the rmse
y_train.mean()
y_test.mean()

# Creates list with same length as y_test, but only full of y_train's mean
y_pred_base = [y_train.mean() for i in range(len(y_test))]

len(y_test)
len(y_pred_base)

# Define root mean squared error function (will use a few times for evaluation)


def rmse(y_true, y_pred):
    print(np.sqrt(mean_squared_error(y_true, y_pred)))


rmse(y_test, y_pred_base)

################################ XGBOOST #######################################
# XGBoost is a tree boosting/ensemble model

# Initialize model with basic hyperparameters
# objective is reg:linear for regression
xg1 = xgb.XGBRegressor(objective='reg:linear',
                       n_jobs=-1,
                       random_state=23)

# Fit model to training data
xg1.fit(X_train, y_train)

# Create predictions
y_pred_xg = xg1.predict(X_test)

# Score
rmse(y_test, y_pred_xg)

# Look at feature importance (good for interpretability and feature selection)
xgb.plot_importance(xg1)
plt.rcParams['figure.figsize'] = [12, 12]
plt.show()
