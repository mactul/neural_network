def save_training(mlp, filename="sauvegarde.txt"):
    """
    mlp: MLPClassifier object
    filename: str
    
    saves the data in a file that can be understood by the load function
    """
    coefs = mlp.coefs_
    coefs = list(coefs)
    for a in range(len(coefs)):
        coefs[a] = list(coefs[a])
        for b in range(len(coefs[a])):
            coefs[a][b] = list(coefs[a][b])
    string = str(coefs)

    biais = mlp.intercepts_
    biais = list(biais)
    for a in range(len(biais)):
        biais[a] = list(biais[a])
    
    string += "\n"+str(biais)

    file = open(filename, "w")
    file.write(string)
    file.close()

print("loading")

#-----------------------------#
#------------DATAS------------#
#-----------------------------#

#put your data here

#loading of a example dataset
from sklearn.datasets import load_breast_cancer 
dataset = load_breast_cancer() 

data = dataset['data']
target = dataset['target']

print("dataset loaded")

#models import
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier



print("start learning")
#mixes the dataset
X_train, X_test, Y_train, Y_test = train_test_split(data, target, random_state=0) 


#creates the network
mlp = MLPClassifier(activation='tanh', max_iter=1000, alpha=0.000000001, random_state=42, hidden_layer_sizes=[10, 10]) 

#training the network
mlp.fit(X_train, Y_train)

#testing the network
print("score: {:.1f}".format(100*mlp.score(X_test, Y_test)), "% de fiabilité") 


#create a file for saving training
save_training(mlp, "sauvegarde.txt")