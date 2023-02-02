import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split

# based on the following explanation: https://towardsdatascience.com/predict-customer-churn-in-python-e8cd6d3aaa7

dataset = pd.read_csv('dataset/BankChurners.csv')
print(dataset.columns)

def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = spacing
        va = 'bottom'
        if y_value < 0:
            space *= -1
            va = 'top'
        label = "{:.1f}%".format(y_value)

    ax.annotate(label,
                (x_value, y_value),
                xytext=(0, space),
                textcoords="offset points",
                ha='center', va=va)

    add_value_labels(ax)
    ax.autoscale(enable=False, axis='both', tight=False)


def analyze_data():
    dataset2 = dataset[['Customer_Age', 'Gender',
                        'Dependent_count', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category', 'Months_on_book',
                        'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']]
    # Histogram:

    fig = plt.figure(figsize=(15, 12))
    plt.suptitle('Histograms of Numerical Columns\n', horizontalalignment="center", fontstyle="normal", fontsize=24,
                 fontfamily="sans-serif")
    for i in range(dataset2.shape[1]):
        plt.subplot(6, 3, i + 1)
        f = plt.gca()
        f.set_title(dataset2.columns.values[i])
        vals = np.size(dataset2.iloc[:, i].unique())
        if vals >= 100:
            vals = 100

        plt.hist(dataset2.iloc[:, i], bins=vals, color='#ec838a')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def show_churn_rate():
    churn_rate = dataset.loc[:, ("Attrition_Flag", "CLIENTNUM")]
    churn_rate["churn_label"] = pd.Series(
        np.where((churn_rate["Attrition_Flag"] == "Existing Customer"), "No", "Yes"))
    sectors = churn_rate.groupby("churn_label")
    churn_rate = pd.DataFrame(sectors["CLIENTNUM"].count())
    churn_rate["Churn Rate"] = (
                                       churn_rate["CLIENTNUM"] / sum(churn_rate["CLIENTNUM"])) * 100
    ax = churn_rate[["Churn Rate"]].plot.bar(title='Overall Churn Rate', legend=True, table=False, grid=False,
                                             subplots=False,
                                             figsize=(12, 7), color='#ec838a', fontsize=15, stacked=False,
                                             ylim=(0, 100))
    plt.ylabel('Proportion of Customers', horizontalalignment="center",
               fontstyle="normal", fontsize="large", fontfamily="sans-serif")
    plt.xlabel('Churn', horizontalalignment="center", fontstyle="normal", fontsize="large", fontfamily="sans-serif")
    plt.title('Overall Churn Rate \n', horizontalalignment="center",
              fontstyle="normal", fontsize="22", fontfamily="sans-serif")
    plt.legend(loc='upper right', fontsize="medium")
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.yticks(rotation=0, horizontalalignment="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    x_labels = np.array(churn_rate[["CLIENTNUM"]])
    plt.show()


analyze_data()
show_churn_rate()

# Filter features to select only numerical ones
filtered_df = dataset[['Attrition_Flag', 'Customer_Age', 'Dependent_count', 'Months_on_book',
                       'Total_Relationship_Count', 'Months_Inactive_12_mon',
                       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']]

classifier = LogisticRegression(random_state=0, penalty='l2', solver='lbfgs', max_iter=100)

X = filtered_df.drop("Attrition_Flag", axis=1).values
y = np.where(filtered_df["Attrition_Flag"] == 'Attrited Customer', 0, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

classifier.fit(x_train, y_train)
# Predict the Test set results
y_pred = classifier.predict(x_test)
# Evaluate Model Results on Test Set:
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=2.0)
results = pd.DataFrame([['Logistic Regression',
                         acc, prec, rec, f1, f2]],
                       columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score'])
print(results)
