
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import numpy as np

def binary_classification(inputs, target, target_names, title, use_pos_label):
    accuracy=[]

    print(f"############{title}############")

    # splitting training and testing data
    input_train, input_test, target_train, target_test = train_test_split(inputs, target, test_size=0.25, random_state=0)

    # Using Decision Tree as a classifier model
    print("Decision Tree ID3")
    model = DecisionTreeClassifier()
    model.fit(input_train, target_train)

    # Predicting test set results
    target_pred = model.predict(input_test)

    # Printing the confusion matrix
    print("\nConfusion matrix")
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)
    accuracy.append(metrics.accuracy_score(target_test,target_pred))
    print(f"Accuracy: {accuracy[-1]}\n")

    # Evaluating fpr, tpr, auc score
    probs = model.predict_proba(input_test)

    fpr1, tpr1, threshold1 = metrics.roc_curve(target_test, probs[:, 0], pos_label=0+use_pos_label)
    roc_auc1 = metrics.auc(fpr1, tpr1)

    # Create a NaiveBayes Classifier model
    print("Naive Bayes Algorithm")
    gnb = GaussianNB()
    gnb.fit(input_train, target_train)

    # Predicting test results
    target_pred = gnb.predict(input_test)

    # Confusion matrix for NaiveBayes Classifier
    print("\nConfusion matrix")
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)
    accuracy.append(metrics.accuracy_score(target_test, target_pred))
    print(f"Accuracy : {accuracy[-1]}\n")

    # evaluating tpr,fpr, threshold
    probs = gnb.predict_proba(input_test)

    fpr2, tpr2, threshold2 = metrics.roc_curve(target_test, probs[:, 0], pos_label=0+use_pos_label)
    roc_auc2 = metrics.auc(fpr2, tpr2)

    """
    # plotting ROC curve
    plt.figure("ROC curve")
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Decision Tree Classifier (AUROC = %0.3f)' % roc_auc1)
    plt.plot(fpr2, tpr2, linestyle='--', color='blue', label='Naive Bayes Classifier (AUROC = %0.3f)' % roc_auc2)
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    #plt.savefig('Binary ROC', dpi=300);
    plt.show()
    """

    #returning accuracies of both classifiers
    return accuracy

def multi_class_classification(inputs, target, target_names, title, use_pos_label):
    accuracy=[]
    print(f"############{title}############")

    # splitting training and testing data
    input_train, input_test, target_train, target_test = train_test_split(inputs, target, test_size=0.25, random_state=0)

    # Using Decision Tree as a classifier model
    print("Decision Tree ID3")
    model = DecisionTreeClassifier()
    model.fit(input_train, target_train)

    # Predicting test set results
    target_pred = model.predict(input_test)

    # Printing the confusion matrix
    print("\nConfusion matrix")
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)
    accuracy.append(metrics.accuracy_score(target_test, target_pred))
    print(f"Accuracy: {accuracy[-1]}\n")

    # evaluating tpr,fpr, threshold
    probs = model.predict_proba(input_test)

    tpr = {}
    fpr = {}
    threshold = {}
    roc_auc = {}
    for i in range(len(target_names)):
        fpr[i], tpr[i], threshold[i] = metrics.roc_curve(target_test, probs[:, i], pos_label=i+use_pos_label)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    """
    # plotting ROC curve
    plot1 = plt.figure(1)
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label=f'Class {target_names[0]} vs Rest (AUROC = %0.3f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label=f'Class  {target_names[1]}vs Rest (AUROC = %0.3f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label=f'Class {target_names[2]} vs Rest (AUROC = %0.3f)' % roc_auc[2])
    plt.title(f'{title} (Decision tree Classifier)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    #plt.savefig('Multiclass ROC', dpi=300)
    """

    # Create a NaiveBayes Classifier model
    print("Naive Bayes Algorithm")
    gnb = GaussianNB()
    gnb.fit(input_train, target_train)

    # Predicting test results
    print("\nConfusion matrix")
    target_pred = gnb.predict(input_test)

    # Confusion matrix for NaiveBayes Classifier
    cm = metrics.confusion_matrix(target_test, target_pred)
    print(cm)
    accuracy.append(metrics.accuracy_score(target_test, target_pred))
    print(f"Accuracy: {accuracy[-1]}\n")

    # evaluating tpr,fpr, threshold
    probs = gnb.predict_proba(input_test)

    tpr = {}
    fpr = {}
    threshold = {}
    roc_auc = {}
    for i in range(len(target_names)):
        fpr[i], tpr[i], threshold[i] = metrics.roc_curve(target_test, probs[:, i], pos_label=i+use_pos_label)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    """
    # plotting ROC curve
    plot2 = plt.figure(2)
    plt.plot(fpr[0], tpr[0], linestyle='-', color='orange',label=f'Class {target_names[0]} vs Rest (AUROC = %0.3f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green',label=f'Class  {target_names[1]}vs Rest (AUROC = %0.3f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue',label=f'Class {target_names[2]} vs Rest (AUROC = %0.3f)' % roc_auc[2])
    plt.title(f'{title}(Naive Bayes Classifier)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    #plt.savefig('Multiclass ROC', dpi=300);
    plt.show()
    """

    return accuracy

import pandas as pd
from sklearn.preprocessing import LabelEncoder

###comparison accuracy
accuracy_dt_with_outlier=[]
accuracy_nb_with_outlier=[]

"""
####Habermans Survival
#Reading dataset
df=pd.read_csv("haberman_survival_data.csv")

#Separating independent and dependent variables
inputs=df.iloc[:, 0:3].values
target=df.iloc[:,3].values
target_names = ['Survived more than 5years', 'Died within 5 years']
title="Habermans Survival"
accuracy_matrix=binary_classification(inputs, target, target_names, title, 1)
accuracy_dt_with_outlier.append(accuracy_matrix[0])
accuracy_nb_with_outlier.append(accuracy_matrix[1])
"""

###Hayes_roth
#Reading dataset
df=pd.read_csv("hayes_roth_data.csv")

#Separating independent and dependent variables
inputs=df.iloc[:, 2:5].values
target=df.iloc[:,5].values
target_names=['1','2','3']

#Classifying dataset
accuracy_matrix=multi_class_classification(inputs,target,target_names,"Hayes Roth",1)
accuracy_dt_with_outlier.append(accuracy_matrix[0])
accuracy_nb_with_outlier.append(accuracy_matrix[1])

"""
###Checking for correlation between variables

dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()"""

###Ionosphere
#Reading dataset
df=pd.read_csv("ionosphere_data.csv")

#Separating independent and dependent variables
inputs=df.iloc[:, 0:33].values
target=df.iloc[:,33].values

#Encoding target columns to 0 and 1 classes
le = LabelEncoder()
target_n=le.fit_transform(target)

#Classifying dataset
target_names = ['good', 'bad']
title="Ionosphere"
accuracy_matrix=binary_classification(inputs, target_n, target_names, title, 0)
accuracy_dt_with_outlier.append(accuracy_matrix[0])
accuracy_nb_with_outlier.append(accuracy_matrix[1])

"""
###Checking for correlation between variables
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()
"""

###IRIS DATASET
#Reading dataset
df=pd.read_csv("iris_data.csv")

#Separating independent and dependent variables
inputs=df.iloc[:, 0:4].values
target=df.iloc[:,4].values

#Encoding target columns to 0 and 1 classes
le = LabelEncoder()
target_n=le.fit_transform(target)
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
accuracy_matrix=multi_class_classification(inputs, target_n,target_names, "Iris", 0)
accuracy_dt_with_outlier.append(accuracy_matrix[0])
accuracy_nb_with_outlier.append(accuracy_matrix[1])

"""
###Checking for correlation between variables
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()
"""

###Wine dataset
#Reading dataset
df=pd.read_csv("wine_data.csv")

#Separating independent and dependent variables
inputs=df.iloc[:, 1:14].values
target=df.iloc[:,0].values

#Classifying dataset
target_names=['class 1','class 2','class 3']
accuracy_matrix=multi_class_classification(inputs,target, target_names,"Wine", 1)
accuracy_dt_with_outlier.append(accuracy_matrix[0])
accuracy_nb_with_outlier.append(accuracy_matrix[1])


"""
###Checking for correlation between variables
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()
"""

###CMC dataset
#Reading dataset
df=pd.read_csv("cmc_data.csv")

#Separating independent and dependent variables
inputs=df.iloc[:, 0:9].values
target=df.iloc[:,9].values
target_names=['No-use','Long-term use','Short-term use']

#Classifying dataset
accuracy_matrix=multi_class_classification(inputs,target,target_names,"Contraceptive Method Choice",1)
accuracy_dt_with_outlier.append(accuracy_matrix[0])
accuracy_nb_with_outlier.append(accuracy_matrix[1])

"""
###Checking for correlation between variables
dataplot = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()
"""

plot10=plt.figure(10)
x=["Hayes_roth","Ionosphere","Iris","Wine","Contraceptive Choice"]
plt.plot(x,accuracy_dt_with_outlier,marker='o',linestyle='solid')
plt.plot(x,accuracy_nb_with_outlier,marker='o',linestyle='solid')
plt.title("Comparison of results of Naive Bayes and Decision Tree")
plt.legend(['Decision Tree','Naive Bayes'], loc='best')
plt.xlabel('Datasets')
plt.ylabel('Accuracy Score')
#plt.show()

######Model accuracy comparison after outlier removal###
accuracy_nb_without_outlier=[]
accuracy_dt_without_outlier=[]

###without outliers
###Hayes_roth
#Reading dataset
df=pd.read_csv("hayes_roth_data.csv")

#Separating independent and dependent variables
z = np.abs(stats.zscore(df))
entries=(z < 3)
inputs_o=df[entries.all(axis=1)]

inputs_o=pd.DataFrame(inputs_o)

inputs=inputs_o.iloc[:, 2:5].values
target=inputs_o.iloc[:,5].values
target_names=['1','2','3']

#Classifying dataset
accuracy_matrix=multi_class_classification(inputs,target,target_names,"Hayes Roth",1)
accuracy_dt_without_outlier.append(accuracy_matrix[0])
accuracy_nb_without_outlier.append(accuracy_matrix[1])

###Ionosphere
#Reading dataset
df=pd.read_csv("ionosphere_data.csv")

#Separating independent and dependent variables
inputs_x=df.iloc[:,0:33]
target_x=df.iloc[:,33]

#Encoding target columns to 0 an
# d 1 classes
le = LabelEncoder()
target_n=le.fit_transform(target_x.values)

#joining input(dataframe) and target(ndarray)
target_x=pd.DataFrame(target_n, columns=['class'])
inputs_x=pd.concat([inputs_x, target_x], axis=1)

z = np.abs(stats.zscore(inputs_x))
entries=(z < 3)
inputs_o=inputs_x[entries.all(axis=1)]

inputs_o=pd.DataFrame(inputs_o)
inputs=inputs_o.iloc[:, 0:33].values
target=inputs_o.iloc[:,33].values

#Classifying dataset
target_names = ['good', 'bad']
title="Ionosphere"
accuracy_matrix=binary_classification(inputs, target, target_names, title, 0)
accuracy_dt_without_outlier.append(accuracy_matrix[0])
accuracy_nb_without_outlier.append(accuracy_matrix[1])

###IRIS DATASET
#Reading dataset
df=pd.read_csv("iris_data.csv")

#Separating independent and dependent variables
inputs_x1=df.iloc[:, 0:4]
target_x1=df.iloc[:,4].values

#Encoding target columns to 0 and 1 classes
le = LabelEncoder()
target_n=le.fit_transform(target_x1)
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#joining input(dataframe) and target(ndarray)
target_x1=pd.DataFrame(target_n, columns=['class'])
inputs_x=pd.concat([inputs_x1, target_x1], axis=1)

z = np.abs(stats.zscore(inputs_x))
entries=(z < 3)
inputs_o=inputs_x[entries.all(axis=1)]

inputs_o=pd.DataFrame(inputs_o)
inputs=inputs_o.iloc[:, 0:4].values
target=inputs_o.iloc[:,4].values

accuracy_matrix=multi_class_classification(inputs, target,target_names, "Iris", 0)
accuracy_dt_without_outlier.append(accuracy_matrix[0])
accuracy_nb_without_outlier.append(accuracy_matrix[1])

###Wine dataset
#Reading dataset
df=pd.read_csv("wine_data.csv")

#Separating independent and dependent variables

z = np.abs(stats.zscore(df))
entries=(z < 3)
inputs_o=df[entries.all(axis=1)]

inputs_o=pd.DataFrame(inputs_o)
inputs=inputs_o.iloc[:, 1:14].values
target=inputs_o.iloc[:,0].values


#Classifying dataset
target_names=['class 1','class 2','class 3']
accuracy_matrix=multi_class_classification(inputs,target, target_names,"Wine", 1)
accuracy_dt_without_outlier.append(accuracy_matrix[0])
accuracy_nb_without_outlier.append(accuracy_matrix[1])

###CMC dataset
#Reading dataset
df=pd.read_csv("cmc_data.csv")

#Separating independent and dependent variables
z = np.abs(stats.zscore(df))
entries=(z < 3)
inputs_o=df[entries.all(axis=1)]

inputs_o=pd.DataFrame(inputs_o)
inputs=inputs_o.iloc[:, 0:9].values
target=inputs_o.iloc[:,9].values


target_names=['No-use','Long-term use','Short-term use']

#Classifying dataset
accuracy_matrix=multi_class_classification(inputs,target,target_names,"Contraceptive Method Choice",1)
accuracy_dt_without_outlier.append(accuracy_matrix[0])
accuracy_nb_without_outlier.append(accuracy_matrix[1])

plot10=plt.figure(11)
x=["Hayes_roth","Ionosphere","Iris","Wine","Contraceptive Choice"]
plt.plot(x,accuracy_dt_without_outlier,marker='o',linestyle='solid')
plt.plot(x,accuracy_nb_without_outlier,marker='o',linestyle='solid')
plt.title("Comparison of results of Naive Bayes and Decision Tree after removal of outliers")
plt.legend(['Decision Tree','Naive Bayes'], loc='best')
plt.xlabel('Datasets')
plt.ylabel('Accuracy Score')

#Comparing Results of Naive Bayes before and after outlier removal
plot11=plt.figure(12)
x=["Hayes_roth","Ionosphere","Iris","Wine","Contraceptive Choice"]
plt.plot(x,accuracy_nb_with_outlier,marker='o',linestyle='solid')
plt.plot(x,accuracy_nb_without_outlier,marker='o',linestyle='solid')
plt.title("Comparison of results of Naive Bayes before and after removal of outlier")
plt.legend(['Before','After'], loc='best')
plt.xlabel('Datasets')
plt.ylabel('Accuracy Score')

#Comparing results of Decision tree before and after outlier removal
plot11=plt.figure(13)
x=["Hayes_roth","Ionosphere","Iris","Wine","Contraceptive Choice"]
plt.plot(x,accuracy_dt_with_outlier,marker='o',linestyle='solid')
plt.plot(x,accuracy_dt_without_outlier,marker='o',linestyle='solid')
plt.title("Comparison of results of Decision Tree before and after removal of outlier")
plt.legend(['Before','After'], loc='best')
plt.xlabel('Datasets')
plt.ylabel('Accuracy Score')
plt.show()