import operator
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print('training_done')

    def predict(self,X_test):
        predictions=[]
        for X in X_test:

            distance={}
            counter=0
            for i in self.X_train:
                distance[counter]=((i[0]-X[0])**2 + (i[1]-X[1])**2) ** 0.5
                counter+=1

            distance=sorted(distance.items(), key=operator.itemgetter(1))
            predictions.append(self.classify(distance[:self.k]))

        return predictions

    def classify(self,distance):
        label=[]

        for idx,_ in distance:
            label.append(self.y_train[idx])

        return (Counter(label).most_common(1)[0][0])





