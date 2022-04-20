from matplotlib.transforms import Transform
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd 

class FraudD:
    def __init__(self, data, type_object_encod="hot_e") -> None:
        self.data = data
        self.type_object_encod = type_object_encod
        self.object_col = [i for i in self.data.columns if self.data[i].dtypes == "object"]

    def split(self):
        data = data.values
        size = data.shape[0]
        train_l = int(size*self.split_precentage)
        return data[:train_l,:], data[train_l:, :]

    def encoding(self):
        for i in self.object_col:
            # classes = self.data[i].value_counts().index
            if self.type_object_encod == "hot_e":
                one_hot = OneHotEncoder(handle_unknown='ignore')
                transform_data = pd.DataFrame(one_hot.fit_transform(self.data[[i]]).toarray())
                self.data = self.data.join(transform_data)
                self.data.rename({0: 'CASH_IN', 1: 'CASH_OUT', 2:'DEBIT', 3:"PAYMENT", 4:"TRANSFER"}, axis=1, inplace=True)
                self.data.drop(i, axis=1, inplace=True)

            elif self.type_object_encod == "labelEncoder":
                label_enocder = LabelEncoder()
                transform_data = label_enocder.fit_transform(self.data[[i]])
                self.data[i] = transform_data
    
    def normlization(self):
        print(self.object_col)
        norm = MinMaxScaler()
        for i in self.data.columns: 
            if i in self.object_col: 
                continue
            else: 
                self.data[i] = norm.fit_transform(self.data[i].to_numpy().reshape(-1,1))

    def data_spliting(self,target,split_precentage):
        total_len = self.data.shape[0]
        train_size = int(total_len * split_precentage)

        train = self.data.loc[:train_size,:]
        test = self.data.loc[train_size:,:]

        print(train)
        test_y = test[target]
        train_y = train[target]

        test.drop(target,inplace=True,axis=1)
        train.drop(target,inplace=True,axis=1)

        return train,train_y,test,test_y

    def logastic_reg(self,train,train_y,test,test_y,iteration):
        log_r = LogisticRegression(solver="liblinear",
                                    max_iter=iteration,
                                    verbose=2,
                                    warm_start=True,
                                    random_state=0).fit(train, train_y)

        pred_score = log_r.score(test,test_y)

        print("Accuracy: ", pred_score , "%")

        return pred_score
    def svm(self,train,train_y,test,test_y,iteration):
        svm_ = SVC(kernel = "sigmoid" ,gamma='auto',verbose=True,max_iter=iteration).fit(train,train_y)

        pred_score = svm_.score(test,test_y)
        print("Accuracy: ", pred_score , "%")

        return pred_score

        