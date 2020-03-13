import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.tree
from sklearn.model_selection import train_test_split

# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')

X = np.array(data[predictors])
Y = np.array(data['Target'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)
print(np.shape(y_train))

def loqueGrafico(X, Y, X_test, Y_test, n):
    n_points = len(Y)
    # esta es la clave del bootstrapping: la seleccion de indices de "estudiantes"
    indices = np.random.choice(np.arange(n_points), n_points)
    new_X = X[indices, :]
    new_Y = Y[indices]
    Arbolito = sklearn.tree.DecisionTreeClassifier(max_depth=n)
    Arbolito.fit(new_X, new_Y)
    f1_train = sklearn.metrics.f1_score(new_Y, Arbolito.predict(new_X))
    f1_test = sklearn.metrics.f1_score(Y_test, Arbolito.predict(X_test))
    importancias_train = Arbolito.feature_importances_
    devuelvo = np.append(f1_train, importancias_train)
    devuelvo = np.append(f1_test, devuelvo)
    return devuelvo
promediosParam = np.ones([14,10])
f1_test = []
f1_train = []
std_test = []
std_train = []
for j in range(1,11):
    Boots = 100
    loqueleo = np.ones([Boots, 16])
    for i in range(Boots):
        loqueleo[i,:] = loqueGrafico(x_train, y_train,x_test, y_test,j)
    f1_train.append(np.mean(loqueleo[:,1]))
    f1_test.append(np.mean(loqueleo[:,0]))
    std_train.append(np.std(loqueleo[:,1]))
    std_test.append(np.std(loqueleo[:,0]))
    promediosParam[:,j-1] = np.mean(loqueleo[:,2:])
    
    
plt.errorbar(range(1,11),f1_test, yerr=std_test, label='test')
plt.errorbar(range(1,11),f1_train, yerr=std_train, label='train')
plt.legend()
plt.savefig("F1_training_test.png")

a = range(1,12)
A = [a,a,a,a,a,a,a,a,a,a,a,a,a,a]
plt.figure(figsize=(12,12))
for i in range(14):
    plt.plot(range(1,11),promediosParam[i,:], label=str(i))
plt.legend()
plt.savefig("features.png")
    
    
    
    