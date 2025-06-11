# SciKit-Learn
Practicing ML using various scikit tools

✅ Ways to Select Features for Model Training
1.	Understand Your Data
	* 	Review data types, missing values, and summary stats:
df.info()
df.describe()
df.head()

2.	Check Correlation (Numeric Features)
	* 	Visualize correlation with the target:
import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

3.	Feature Importance (Tree-Based Models)
	* 	Use a RandomForest to find important features:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
feature_names = X.columns
sorted_indices = importances.argsort()[::-1]
plt.barh(feature_names[sorted_indices], importances[sorted_indices])
plt.xlabel("Feature Importance")
plt.show()

4.	Univariate Feature Selection
	* 	Use statistical tests to pick top features:
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

5.	Remove Irrelevant/Redundant Features
	* 	Drop:
	* 	Features with many missing values.
	* 	ID columns or constants.
	* 	Features highly correlated with each other.

6.	Recursive Feature Elimination (RFE)
	*  Automatically remove weakest features:
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(X, y)
selected_features = X.columns[fit.support_]

7.	Use Domain Knowledge
	* 	Include features known to influence the target even if stats don’t show strong correlation.
