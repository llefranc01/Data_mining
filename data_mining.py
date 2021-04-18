import pandas as pd

'for clustering'
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from sklearn.feature_selection import mutual_info_classif
import seaborn as sns

'for classifying'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

'for accuracy analysis'
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('processed.csv')
'-----------------describe data--------------------------------------'

print(data.describe(include='all'))

correlation_heat = data.corr()
sns.heatmap(correlation_heat, xticklabels=['College', 'Income', 'Overage', 'Leftover', 'House', 'Handset', 'Over15', 'AvgCallDur', 'RepSatis', 'RepUsage', 'ConsidChange', 'Leave'], yticklabels=['College', 'Income', 'Overage', 'Leftover', 'House', 'Handset', 'Over15', 'AvgCallDur', 'RepSatis', 'RepUsage', 'ConsidChange', 'Leave'])
plt.show()


def gain_chart():
    data = pd.read_csv('processed.csv')
    df_leave = data['LEAVE']
    data.drop(columns=['LEAVE'], inplace=True)
    res = list(zip(list(data.columns), mutual_info_classif(data, df_leave, discrete_features=True)))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    (labels, info_gain) = zip(*res)
    plt.bar(['H', 'I', 'O', 'HP', 'O_15', 'L', "Aver", "RS", "C", "CCP", "RU"], list(info_gain))
    for i, v in enumerate(list(info_gain)):
        plt.text(i - 0.45, v + 0.01, str('%.3f'%(v)))
    plt.xlabel('Attributes', fontweight='bold')
    plt.ylabel('Information Gain', fontweight='bold')
    plt.title('Information gain display')
    plt.show()


gain_chart()

'--------------------------------------------------------------------'

X = data.drop(columns=['LEAVE'])
y = data['LEAVE']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=True)

model = DecisionTreeClassifier(max_depth=10)
model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print('accuracy of DecisionTree')
print(accuracy_score(y_test, y_predict))

model2 = KNeighborsClassifier(n_neighbors=3)
model2 = model2.fit(X_train, y_train)

y_predict = model2.predict(X_test)
print('accuracy of K nearest neighbor')
print(accuracy_score(y_test, y_predict))

'------------------------------------------------ clustering-----------'

print('CUSTOMER SEGMENTATION WITH K MEANS')

def find_best_k():
    
    k_clusters = 10

    data = pd.read_csv('processed.csv')
    data = data.to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    print("starting k-means training")
    SSE = []
    sil = []
    for k in range(2, k_clusters):
        print('kmeans = {}'.format(k))
        kmeans = KMeans(n_jobs=-1, n_clusters=k, init='k-means++')
        kmeans.fit(data_scaled)
        SSE.append(kmeans.inertia_)
        sil.append(silhouette_score(data_scaled, kmeans.labels_, metric='euclidean'))

    frame = pd.DataFrame({'Cluster': range(2, k_clusters), 'SSE': SSE})
    plt.figure(figsize=(12, 6))
    plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    plt.xticks(range(1,k_clusters))
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum Squared Error')

    plt.show()

    frame = pd.DataFrame({'Cluster': range(2, k_clusters), 'Silhouette': sil})
    plt.figure(figsize=(12, 6))
    plt.plot(frame['Cluster'], frame['Silhouette'], marker='o')
    plt.xticks(range(1, k_clusters))
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette')

    plt.show()


def analyze_clusters(k):
    
    data = pd.read_csv('processed.csv')
    data = data.to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_jobs=-1, n_clusters=k, init='k-means++')
    kmeans.fit(data_scaled)
    labels = kmeans.labels_

    clusters = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i, cluster in enumerate(labels):
        clusters[cluster].append(data[i].tolist())
    df_clusters = []
    for i in range(5):
        df_clusters.append(pd.DataFrame(clusters[i]))

    for i, df in enumerate(df_clusters):
        print('CLUSTER {}'.format(i))
        print('------------------------------------')
        print(df.describe(include='all'))
        print('------------------------------------')
        print()

find_best_k()

analyze_clusters(5)

'-------------------------------------------------------------------'