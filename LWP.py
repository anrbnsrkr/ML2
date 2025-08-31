import numpy as np
import sklearn.metrics as matrices


class LWP :
    avg = np.zeros((1,1))
    count = np.zeros((1,1))
    classes = 0
    vecLen = 0
    def __init__(self,numClasses : int,veclen:int):
        self.avg = np.zeros((numClasses,veclen))
        self.classes = numClasses
        self.count = np.zeros(numClasses,dtype=np.float64)
        self.vecLen = veclen
    def fit(self,x_train,y_train):
        if x_train.ndim != 2:
            print(f"Invalid X_train ndim = {x_train.ndim} expected = 2")
            return None
        if x_train.shape[1] != self.vecLen:
            print(f"Invalid vectro size ndim = {x_train.shape[1]} expected = {self.vecLen}")
            return None
        for i in range(x_train.shape[0]):
            y = y_train[i]
            c = self.count[y]
            self.count[y] += 1
            self.avg[y] = ((c/self.count[y]) * self.avg[y]) + (x_train[i]/self.count[y])

    def predict(self,x_pred):
        if x_pred.ndim != 2:
            print(f"Invalid X_train ndim = {x_pred.ndim} expected = 2")
            return None

        if x_pred.shape[1] != self.vecLen:
            print(f"Invalid vectro size ndim = {x_pred.shape[1]} expected = {self.vecLen}")

        y_pred = np.empty((x_pred.shape[0],self.classes))

        for i in range(x_pred.shape[0]):
            div = 0.0
            for j in range(self.classes):
                d = np.linalg.norm(x_pred[i] - self.avg[j])
                y_pred[i][j] = np.exp(-d)
                div += y_pred[i][j]

            if div >= 0:
                y_pred[i] /= div

        return y_pred

    def predToAbs(self, y_pred):
        y_pred_abs = np.empty(len(y_pred),dtype=np.int32)
        for i in range(len(y_pred)):
            y_pred_abs[i] = np.argmax(y_pred[i])

        return y_pred_abs

    def evaluate(self, x_test, y_test):
        if x_test.ndim != 2:
            print(f"Invalid X_train ndim = {x_test.ndim} expected = 2")
            return None

        if x_test.shape[1] != self.vecLen:
            print(f"Invalid vectro size ndim = {x_test.shape[1]} expected = {self.vecLen}")
        y_pred = self.predict(x_test)
        y_pred_abs = self.predToAbs(y_pred)
        correct = 0
        for i in range(y_pred_abs.shape[0]):
            if y_pred_abs[i] == y_test[i]:
                correct += 1
        return correct/len(y_pred_abs)

if __name__ == '__main__':
    lwp = LWP(2,2)
    lwp.fit(np.array([[2,2],[-2,-2],[1,1],[-1,-1]]),np.array([0,1,0,1]))
    print(lwp.avg)
    pred = lwp.predict(np.array([[1, 1],[1, 2]]))
    print(pred)
    acc = lwp.evaluate(np.array([[1, 1],[1, 2]]), np.array([0,1])  )
    print(acc)
