

import pandas as pd
import numpy as np

import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8, 5)

from sklearn.ensemble import BaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, f1_score

from sklearn.preprocessing import LabelBinarizer


from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import umap

eclipsejdt = pd.read_csv('eclipsejdtcore_.csv')
eclipsepde = pd.read_csv('eclipsepdeui_.csv')
equinox = pd.read_csv('equinoxframework_.csv')
lucene = pd.read_csv('lucene_.csv')
mylyn = pd.read_csv('mylyn_.csv')

print("NaNs in eclipse_jdt", np.sum(np.sum(eclipsejdt.isna(), axis=0)) )
print("NaNs in eclipse_pdt", np.sum(np.sum(eclipsepde.isna(), axis=0)) )
print("NaNs in equinox", np.sum(np.sum(equinox.isna(), axis=0)) )
print("NaNs in lucene", np.sum(np.sum(lucene.isna(), axis=0)) )
print("NaNs in mylyn", np.sum(np.sum(mylyn.isna(), axis=0)) )

eclipsepde.dropna(axis = 1, inplace=True)
equinox.dropna(axis = 1, inplace=True)
lucene.dropna(axis = 1, inplace=True)
mylyn.dropna(axis = 1, inplace=True)
na = eclipsejdt.isnull().sum()
print(na)

print("Data Shapes:", eclipsejdt.shape, eclipsepde.shape, equinox.shape, lucene.shape, mylyn.shape)
df = pd.concat([eclipsejdt, eclipsepde, equinox, lucene, mylyn], ignore_index=True)
df.columns = df.columns.str.replace(' ', '')
print("Full dataframe shape:",df.shape, '\n')

print("Data Shapes:", eclipsejdt.shape, eclipsepde.shape, equinox.shape, lucene.shape, mylyn.shape)
df = pd.concat([eclipsejdt, eclipsepde, equinox, lucene, mylyn], ignore_index=True)
df.columns = df.columns.str.replace(' ', '')
print("Full dataframe shape:",df.shape, '\n')
print("Predictors:")
for name in df.columns.values[1:18].tolist():
    print(name, end=', ')
print("\n\nPredictable:", df.columns.values[18])
print(df.head(5))

submission_file_path = "project.csv"
df.to_csv(submission_file_path, index=False)
print(df.describe())

df = df.drop(['nonTrivialBugs', 'majorBugs', 'criticalBugs', 'highPriorityBugs'], axis=1)
df = df.sample(frac=1.0)

x=df.iloc[:, 1:-2]
y=df['bugs']
print("x:", x.shape)
print("y:", y.shape)

unique, counts = np.unique(y, return_counts=True)
print("Classes:", unique.tolist())
print("Counts:", counts.tolist())

plt.bar(unique, counts, color=['g', 'orange', 'r'], alpha=0.7)
plt.title("#Bugs VS Occurrences")
plt.xticks(range(len(unique)))
plt.ylabel("Occurrences")
plt.xlabel("# Bugs");

"""Make Classes [0, 1, 2] Bugs"""

y = y.clip(upper=2)

"""Class Balance"""

unique, counts = np.unique(y, return_counts=True)
print("Classes:", unique.tolist())
print("Counts:", counts.tolist())

plt.bar(unique, counts, color=['g', 'orange', 'r'], alpha=0.7)
plt.title("Bugs VS Occurrences")
plt.xticks(range(len(unique)))
plt.ylabel("Occurrences")
plt.xlabel("Bugs");

"""more EDA"""

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.5)
print("Train:", X_train.shape, y_train.shape,
      "Test:", X_test.shape, y_test.shape,
      "Cross Validation", X_cv.shape, y_cv.shape)

unique, counts = np.unique(y_train, return_counts=True)
print("Classes:", unique.tolist())
print("Counts:", counts.tolist())

plt.bar(unique, counts, color=['g', 'orange', 'r'], alpha=0.7)
plt.title("Bugs VS Occurrences")
plt.xticks(range(len(unique)))
plt.ylabel("Occurrences")
plt.xlabel("Bugs");

"""scaling features"""

X_train_scaled = pd.DataFrame(StandardScaler().fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(StandardScaler().fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)

data_for_viz = X_train_scaled.copy()
data_for_viz_unscaled = X_train.copy()

data_for_viz['Bugs'] = y_train.copy().tolist()
data_for_viz_unscaled['Bugs'] = y_train.copy().tolist()

fig, ax = plt.subplots(figsize=(10,8))

corr = data_for_viz.corr()
mask = np.triu(corr)
sns.heatmap(corr, vmin=-1, vmax=1, center= 0, cmap= 'Pastel2', linewidths=3, ax=ax, mask=mask);

"""Feature importance with Lasso regression"""

lasso = Lasso()
lasso.fit(X_train,y_train)
coef = pd.Series(lasso.coef_, index = X_train.columns)

print("Discarded Features:", np.sum(lasso.coef_==0), "out of", len(X_train.columns))

coef = coef[coef != 0]

coef.sort_values().plot(kind='barh', cmap="Pastel2");

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train);
coef_rfc = pd.Series(rfc.feature_importances_, index = X_train.columns)

preds = rfc.predict(X_train)
print("accuracy score:", accuracy_score(y_train, preds))

data_for_viz_unscaled.head()

for_pair_plot = data_for_viz_unscaled[['rfc', 'cbo', 'fanOut','wmc', 'numberOfLinesOfCode', 'Bugs']]
pairplot = sns.pairplot(for_pair_plot, hue="Bugs", vars=['rfc', 'cbo', 'fanOut', 'wmc', 'numberOfLinesOfCode']);
pairplot.fig.suptitle("Scatter w.r.t classes: 0, 1, > 2 Bugs", y=1.02);

"""models

"""

X_train_scaled = pd.DataFrame(StandardScaler().fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(StandardScaler().fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)

print("Train:", X_train.shape, y_train.shape,
      "Test:", X_test.shape, y_test.shape,
      "Cross Validation", X_cv.shape, y_cv.shape)

"""multi class"""

x=df.iloc[:, 1:-2]

y_multi = df['bugs']
y_multi = y_multi.clip(upper=2)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(x, y_multi, test_size=0.3, random_state=42)

scaler = StandardScaler()

X_train_m = scaler.fit_transform(X_train_m)
X_test_m = scaler.transform(X_test_m)

pca = PCA(n_components=10)
X_train_m_pca = pca.fit_transform(X_train_m)
X_test_m_pca = pca.transform(X_test_m)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, binary=True):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='binary' if binary else 'weighted')
    cm = confusion_matrix(y_test, preds)

    print(f"\n{name} Results:")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print("Confusion Matrix:\n", cm)
    print(" ")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

evaluate_model("Random Forest | Multi", RandomForestClassifier(), X_train_m, y_train_m, X_test_m, y_test_m, binary=False)

evaluate_model("KNN | Multi", KNeighborsClassifier(), X_train_m, y_train_m, X_test_m, y_test_m, binary=False)

evaluate_model("Bagging | Multi", BaggingClassifier(), X_train_m, y_train_m, X_test_m, y_test_m, binary=False)

evaluate_model("Bagging | Multi | PCA", BaggingClassifier(), X_train_m_pca, y_train_m, X_test_m_pca, y_test_m, binary=False)

evaluate_model("Dummy | Multi", DummyClassifier(strategy='most_frequent'), X_train_m, y_train_m, X_test_m, y_test_m, binary=False)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_m)
preds_kmeans = kmeans.predict(X_test_m)

cm_kmeans = confusion_matrix(y_test_m, preds_kmeans)
print("\nKMeans | Multi Results (Unsupervised):")
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Purples')
plt.title('KMeans Confusion Matrix')
plt.xlabel('Predicted Cluster')
plt.ylabel('True Class')
plt.show()

"""binary class"""

y_binary = df['bugs']
y_binary = y_binary.clip(upper=1)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(x, y_binary, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_b = scaler.fit_transform(X_train_b)
X_test_b = scaler.transform(X_test_b)

"""neural network"""

evaluate_model("Neural Network | Binary", MLPClassifier(max_iter=500), X_train_b, y_train_b, X_test_b, y_test_b)

"""AdaBoost"""

evaluate_model("AdaBoost | Binary", AdaBoostClassifier(), X_train_b, y_train_b, X_test_b, y_test_b)

"""svm"""

evaluate_model("SVM | Binary", SVC(), X_train_b, y_train_b, X_test_b, y_test_b)

"""bagging classifier"""

evaluate_model("Bagging | Binary", BaggingClassifier(), X_train_b, y_train_b, X_test_b, y_test_b)

"""dummy"""

evaluate_model("Dummy | Binary", DummyClassifier(strategy='most_frequent'), X_train_b, y_train_b, X_test_b, y_test_b)

def plot_roc_curve(model, X_test, y_test, name, use_decision_function=False):
    if use_decision_function:
        y_scores = model.decision_function(X_test)
    else:
        y_scores = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
    return auc_score

models_roc = {
    "Neural Net": MLPClassifier(max_iter=500),
    "AdaBoost": AdaBoostClassifier(),
    "SVM": SVC(probability=True),
    "Bagging": BaggingClassifier(),
    "Dummy": DummyClassifier(strategy='most_frequent')
}

plt.figure(figsize=(10, 7))
for name, model in models_roc.items():
    model.fit(X_train_b, y_train_b)
    if hasattr(model, "predict_proba"):
        plot_roc_curve(model, X_test_b, y_test_b, name)
    elif hasattr(model, "decision_function"):
        plot_roc_curve(model, X_test_b, y_test_b, name, use_decision_function=True)

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal baseline
plt.title("ROC Curve Comparison (Binary Classifiers)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

models = {
    "Neural Net": MLPClassifier(max_iter=500),
    "AdaBoost": AdaBoostClassifier(),
    "SVM": SVC(probability=True),
    "Bagging": BaggingClassifier(),
    "Dummy": DummyClassifier(strategy='most_frequent')

}
models_multi = {
    "Random Forest | Multi": RandomForestClassifier(),
    "KNN | Multi": KNeighborsClassifier(),
    "Bagging | Multi": BaggingClassifier(),
    "Dummy | multi": DummyClassifier(strategy='most_frequent')
}

results = []
for name, model in models.items():
    model.fit(X_train_b, y_train_b)
    preds = model.predict(X_test_b)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_b)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test_b)
    else:
        y_scores = preds
    acc = accuracy_score(y_test_b, preds)
    f1 = f1_score(y_test_b, preds)
    auc = roc_auc_score(y_test_b, y_scores)
    fpr, tpr, _ = roc_curve(y_test_b, y_scores)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1, "AUC": auc})

    cm = confusion_matrix(y_test_b, preds)
    print(f"\n{name} Confusion Matrix:\n", cm)

results_df = pd.DataFrame(results)
print("\n Model Performance Summary:")
print(results_df.sort_values(by="AUC", ascending=False).round(4))

for name, model in models_multi.items():
    model.fit(X_train_m, y_train_m)
    preds = model.predict(X_test_m)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_m)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test_m)
    else:
        lb = LabelBinarizer()
        y_scores = lb.fit_transform(preds)
        if y_scores.shape[1] == 1:
            y_scores = np.hstack([1 - y_scores, y_scores])

    acc = accuracy_score(y_test_m, preds)
    f1 = f1_score(y_test_m, preds, average='weighted')

    auc = roc_auc_score(y_test_m, y_scores, multi_class='ovo')

    n_classes = len(np.unique(y_test_m))
    fpr, tpr = dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_m, y_scores[:, i], pos_label=i)

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr = all_fpr
    tpr = mean_tpr

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1, "AUC": auc})

    cm = confusion_matrix(y_test_m, preds)
    print(f"\n{name} Confusion Matrix:\n", cm)

results_df = pd.DataFrame(results)
print("\n Model Performance Summary:")
print(results_df.sort_values(by="AUC", ascending=False).round(4))

"""With Feature Selection(by lasso )

- rfc
- cbo
- wmc
- numberOfLinesOfCode
- lcom
- numberOfAttributes
- numberOfAttributesInherited
"""

results_f = []

selected_features = [
    'rfc', 'cbo', 'wmc', 'numberOfLinesOfCode',
    'lcom', 'numberOfAttributes', 'numberOfAttributesInherited'
]

X_f = df[selected_features]
y_multi = df['bugs'].clip(upper=2)
y_binary = df['bugs'].clip(upper=1)

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def evaluate_model(name, model, X_train, y_train, X_test, y_test, binary=True):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='binary' if binary else 'weighted')
    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)
        else:
            y_scores = model.decision_function(X_test)

        if binary:
            auc = roc_auc_score(y_test, y_scores[:, 1])
        else:
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            auc = roc_auc_score(y_test_bin, y_scores, multi_class='ovr')
    except:
        auc = np.nan

    results_f.append({"Model": name, "Accuracy": acc, "F1 Score": f1, "AUC": auc})

X_train_bf, X_test_bf, y_train_bf, y_test_bf = prepare_data(X_f, y_binary)
evaluate_model("Neural Network | Binary | FS", MLPClassifier(max_iter=500), X_train_bf, y_train_bf, X_test_bf, y_test_bf)
evaluate_model("AdaBoost Classifier | Binary | FS", AdaBoostClassifier(), X_train_bf, y_train_bf, X_test_bf, y_test_bf)
evaluate_model("Support-Vector Machine | Binary | FS", SVC(probability=True), X_train_bf, y_train_bf, X_test_bf, y_test_bf)
evaluate_model("Bagging Classifier | Binary | FS", BaggingClassifier(), X_train_bf, y_train_bf, X_test_bf, y_test_bf)
evaluate_model("Dummy Classifier | Binary | FS", DummyClassifier(strategy='most_frequent'), X_train_bf, y_train_bf, X_test_bf, y_test_bf)

X_train_mf, X_test_mf, y_train_mf, y_test_mf = prepare_data(X_f, y_multi)
evaluate_model("Random Forest | Multi | FS", RandomForestClassifier(), X_train_mf, y_train_mf, X_test_mf, y_test_mf, binary=False)
evaluate_model("K-Nearest Neighbor | Multi | FS", KNeighborsClassifier(), X_train_mf, y_train_mf, X_test_mf, y_test_mf, binary=False)
evaluate_model("Bagging Classifier | Multi | FS", BaggingClassifier(), X_train_mf, y_train_mf, X_test_mf, y_test_mf, binary=False)
evaluate_model("Dummy Classifier | Multi | FS", DummyClassifier(strategy='most_frequent'), X_train_mf, y_train_mf, X_test_mf, y_test_mf, binary=False)

print("\\n Model Performance Using Feature Selection:")

results_df2 = pd.DataFrame(results_f)
print(results_df2.sort_values(by="Model").to_string(index=False))