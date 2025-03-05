from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib


url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
data = pd.read_csv(url)

data = data[['Latitude', 'Longitude', 'MedInc']]

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

print(data.head())
print(X_train.shape)
print(X_test.shape)

kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train)

X_train['Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
plt.scatter(X_train['Longitude'], X_train['Latitude'], c=X_train['Cluster'], cmap='viridis')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters de Casas')
plt.colorbar(label='Cluster')
plt.show()

print(X_train.head())


X_test['Cluster'] = kmeans.predict(X_test)


plt.figure(figsize=(10, 6))
plt.scatter(X_train['Longitude'], X_train['Latitude'], c=X_train['Cluster'], cmap='viridis', alpha=0.6, label='Training Data')
plt.scatter(X_test['Longitude'], X_test['Latitude'], c=X_test['Cluster'], cmap='viridis', marker='x', label='Test Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters de Casas con Datos de Prueba')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()

print(X_test.head())


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train[['Latitude', 'Longitude', 'MedInc']], X_train['Cluster'])

y_pred = clf.predict(X_test[['Latitude', 'Longitude', 'MedInc']])

print("Classification Report:")
print(classification_report(X_test['Cluster'], y_pred))

joblib.dump(kmeans, 'kmeans_model.pkl')

joblib.dump(clf, 'classification_model.pkl')

print("Modelos guardados en 'kmeans_model.pkl' y 'classification_model.pkl'")

