#meta:tag=hide

def get_iris(test_size = 0.2, random_state = 42):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    data = train_test_split(X, y, test_size=test_size, 
                            random_state=random_state)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    return (X, y, X_train, y_train, X_test, y_test, iris)
