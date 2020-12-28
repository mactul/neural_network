import math #if you can't use this import, remove it and rewrite tanh function and logistic function

Activation = int #define a type can be passed to predict, int is the return value of the function


def mat_dim(mat:list) -> list:
    """
    mat: array 2D
    return: tupple 2 values col*row

    returns a pair of column*line values representing the size of the matrix.
    """
    if type(mat[0]) == list:
        return ( len(mat), len(mat[0]) )
    else:
        return ( 1, len(mat) )

def mult_mat(mat1:list, mat2:list) -> list:
    """
    mat1, mat2: array 2D
    return: array 2D

    returns the mathematical product of 2 matrixes
    In maths, matrixes multiply line*column
    """
    d1 = mat_dim(mat1)
    d2 = mat_dim(mat2)
    if d1[1] != d2[0]:
        raise ValueError("the 2 size matrixes", d1, "and", d2, "are not multipliable\nReminder: matrix multiplication is not commutative.")

    result = []

    for row in range(d1[0]):
        ligne = []
        for row2 in range(d2[1]):
            cell = 0
            for col in range(d1[1]):
                cell += mat1[row][col]*mat2[col][row2]
            ligne += [cell]
        result += [ligne]

    return result


def predict(e:list, w:list, p:list, activ:Activation):
    """
    e: list 1D
    w: array 3D
    p: array 2D
    activ: function

    Returns a list representing the network prediction.
    Passes a list of entries, a table of coefficients calculated by sklearn, a table of bias coefficients calculated by sklearn and the activation function (relu, tanh, etc...) and the function determines the shape of the network and calculates the prediction.
    The returned value is an output list produced by the last neurons, it is up to you to interpret it according to the shape of the targets you have used.
    """
    for layer in range(len(w)):
        e = mult_mat([e],w[layer])[0]
        for i in range(len(e)):
            e[i] = activ(p[layer][i]+e[i])
    return e


#------------------------------#
#-----ACTIVATION FUNCTIONS-----#
#------------------------------#

def identity(x:int) -> int:
    """
    no-op activation
    return set: [-infinite; +infinite]

          |  /   
          | /    
    ______|/_____
         /|      
        / |      
       /  |      
    """
    return x

def relu(x:int) -> int:
    """
    the rectified linear unit function
    return set: [0; +infinite]

          |  /   
          | /    
    ______|/_____
    ------|      
          |      
          |      
    """
    if x<0:
        x = 0
    return x

def logistic(x:int) -> int:
    """
    the logistic sigmoid function
    return set: [0; 1]
    """
    return 1 / (1 + math.exp(-x))

def tanh(x:int) -> int:
    """
    the hyperbolic tan function
    return set: [-1; 1]

          |   ___  
          | _-   
    ______|/_____
        _/|      
    ___-  |      
          |      
    """
    return math.tanh(x)




def load(filename:str="sauvegarde.txt") -> (list, list):
    """
    extract coefs and bias from the file
    """
    file = open("sauvegarde.txt", "r")
    contenu = file.read().split('\n')
    file.close()

    w = eval(contenu[0])
    p = eval(contenu[1])

    return w, p


if __name__ == "__main__":
    #import the same dataset that train the network
    from sklearn.datasets import load_breast_cancer 
    dataset = load_breast_cancer() 

    #take one of the multiples datas
    data = dataset['data']
    e = list(data[100])

    #load the network
    w, p = load("sauvegarde.txt")

    #make a prediction (here, a number close to -1 represents a begenin tumour and a number close to 1 represents a malignant tumour)
    print(predict(e,w,p,tanh))