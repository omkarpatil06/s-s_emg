from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# This is the same as doing Leave-One-Out-Cross-Validation.

def logistic_regression_search(param_grid, X, y, groups):
    log_reg = LogisticRegression()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best Logistic Regression Parameters: {grid_search.best_params_}')

def knearest_neighbors_search(param_grid, X, y, groups):
    knn = KNeighborsClassifier()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best K Neighbors Classifier Parameters: {grid_search.best_params_}')

def kmeans_search(param_grid, X, y, groups):
    kmeans = KMeans()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best K Means Classifier Parameters: {grid_search.best_params_}')

def svc_search(param_grid, X, y, groups):
    svc = SVC()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best SVC Classifier Parameters: {grid_search.best_params_}')

def dtc_search(param_grid, X, y, groups):
    dtc = DecisionTreeClassifier()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best Decision Tree Classifier Parameters: {grid_search.best_params_}')

def rf_search(param_grid, X, y, groups):
    rf = RandomForestClassifier()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best Random Forest Classifier Parameters: {grid_search.best_params_}')

def mlp_search(param_grid, X, y, groups):
    mlp = MLPClassifier()
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best Multi-Layer Perceptron Classifier Parameters: {grid_search.best_params_}')

def xgboost_search(param_grid, X, y, groups):
    xgb_clf = XGBClassifier(objective='binary:logistic', random_state=42)
    gkf = GroupKFold(n_splits=30)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=gkf, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    print(f'Best XGBoost Classifier Parameters: {grid_search.best_params_}')