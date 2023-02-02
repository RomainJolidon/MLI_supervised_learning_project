import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing, feature_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import time

inputs = np.load('dataset/inputs.npy')
labels = np.load('dataset/labels.npy')
dataset = np.c_[inputs, labels]

print('prediction of the amount of electricity produced')

cols = [str(i) for i in range(0, 100)]
cols.append("Y")

df = pd.DataFrame(dataset, columns=cols)
#print(df.head())


def plot_dataset():
    x = "Y"
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(x, fontsize=20)

    # distribution
    ax[0].title.set_text('distribution')
    variable = df[x].fillna(df[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[(variable > breaks[0]) & (variable < breaks[10])]
    sns.histplot(variable, kde=True, ax=ax[0])

    des = df[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda v: str(v))
    box = '\n'.join(("min: " + des["min"], "25%: " + des["25%"], "mean: " + des["mean"], "75%: " + des["75%"],
                     "max: " + des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right",
               bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    # boxplot
    ax[1].title.set_text('outliers (log scale)')
    tmp_df = pd.DataFrame(df[x])
    tmp_df[x] = np.log(tmp_df[x])
    tmp_df.boxplot(column=x, ax=ax[1])
    plt.show()


plot_dataset()

df_train, df_test = train_test_split(df, test_size=0.3)

# scale X
scalerX = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X = scalerX.fit_transform(df_train.drop("Y", axis=1))
df_scaled = pd.DataFrame(X, columns=df_train.drop("Y", axis=1).columns, index=df_train.index)
# scale Y
scalerY = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
df_scaled["Y"] = scalerY.fit_transform(df_train["Y"].values.reshape(-1, 1))

print(df_scaled.head())

# Feature selection (useless for now, because we doesn't know what value is what feature)
X = df_train.drop("Y", axis=1).values
y = df_train["Y"].values
feature_names = df_train.drop("Y", axis=1).columns

selector = feature_selection.SelectFromModel(estimator=Ridge(alpha=1.0, fit_intercept=True), max_features=10).fit(X, y)
regularization_selected_features = feature_names[selector.get_support()]

X_names = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', "GarageCars"]
X_train = df_train[X_names].values
y_train = df_train["Y"].values
X_test = df_test[X_names].values
y_test = df_test["Y"].values
