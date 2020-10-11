

import pandas as pd
import treePlotter
import numpy as np

class DecisionClassifier(object):
    
    def __init__(self, max_depth = 7, criterion = "id3", eps = 1e-2):
        
        if not isinstance(max_depth, int) or not max_depth > 0:
            raise Exception("max_depth need to bigger than 0 and the type of it must be int.Please check it.")
        if criterion not in ("id3", "c4.5", "gini"):
            raise Exception("criterion must set 'id3' or 'c4.5' or 'gini'.The type of it is str.Please check it.")
        if not isinstance(eps, float):
            raise Exception("The type of eps must be float,please check it.")
        self.max_depth = max_depth
        self.criterion = criterion
        self.eps = eps

    def fit(self, X, y, show_graph = False):
     
        all_features = list(X.columns)
        
        self.__all_feature_dict = dict([( feature, X[feature].unique() ) for feature in all_features])
        
        self.tree = self.__createTree(X = X, y = y, candidate_features = all_features)
        
        
        if show_graph == True:
            treePlotter.createPlot(self.tree)
        

    def __createTree(self, X, y, candidate_features, cur_depth = 1):
        
       
       
        if cur_depth >= self.max_depth:
            return y.iloc[np.argmax(y.value_counts())]
        
        if y.unique().shape[0] == 1:
            return y.iloc[0]
      
        if candidate_features == []:
            return y.iloc[np.argmax(y.value_counts())]
       
        best_split_feature, biggest_gain = self.__select_best_split_feature(X, y, candidate_features)
       
        if biggest_gain < self.eps:
            return y.iloc[np.argmax(y.value_counts())]
        
        tree = {best_split_feature: {}}
       
        for best_feature_value in self.__all_feature_dict[best_split_feature]:
       
            index = X[ X[best_split_feature] == best_feature_value ].index

           
            if y[index].shape[0] == 0:
                tree[best_split_feature][best_feature_value] = y.iloc[np.argmax(y.value_counts())]
            
            
            else:
                candidate_features_copy = candidate_features.copy()
                candidate_features_copy.remove(best_split_feature)
                tree[best_split_feature][best_feature_value] = self.__createTree(X = X.loc[index], y = y[index], candidate_features = candidate_features_copy, cur_depth = cur_depth + 1)
        return tree

    def __select_best_split_feature(self, X, y, candidate_features):
       
       
        if self.criterion == "id3":
            
            init_purity = self.__calculate_purity(y)
          
            purity_gain = {}
           
            for feature in candidate_features:
               
                purity_fix_feature = 0
                
                groups = y.groupby(X[feature], axis = 0)
                
                for name, group in groups:
                        
                        purity_fix_feature += (group.shape[0] / y.shape[0])*self.__calculate_purity(group)
                
                purity_gain[feature] = init_purity - purity_fix_feature

           
            pruity_gain = pd.Series(purity_gain)
            
            return pruity_gain.index[np.argmax(pruity_gain)] , pruity_gain.max()

        
        if self.criterion == "c4.5":
            init_purity_ratio = self.__calculate_purity(y)
           
            purity_ratio_gain = {}
          
            for feature in candidate_features:
                
                purity_ratio_fix_feature = 0
                
                feature_purity = 0
                
                groups = y.groupby(X[feature], axis = 0)
               
                for name, group in groups:
                    
                    purity_ratio_fix_feature += (group.shape[0] / y.shape[0])*self.__calculate_purity(group)
                    
                    feature_purity += group.shape[0] / y.shape[0]
                
                purity_ratio_gain[feature] = (init_purity_ratio - purity_ratio_fix_feature) / feature_purity
            
           
            pruity_ratio_gain = pd.Series(purity_ratio_gain)
            
            return pruity_ratio_gain.index[np.argmax(pruity_ratio_gain)] , pruity_ratio_gain.max()


    def __calculate_purity(self, y):
        """
        y ： The label set which you want to calculate the purity of it. 
            dtype: Series
        """
       
        prob =  y.value_counts()
       
        if self.criterion == "id3" or "c4.5":
            
            return prob.apply(lambda x: -(x/y.shape[0])*np.log2(x/y.shape[0])).sum()

    def __predict_single_sample(self, x):
       
        tree = self.tree
        while isinstance(tree, dict):
           
            feature = list(tree.keys())[0]
            
            tree = list(tree.values())[0].get(x[feature])
        return tree
            
            

    def predict(self, X):
      
       
        predict_label = []
        for _, row in X.iterrows():
            predict_label.append(self.__predict_single_sample(row))
        return pd.Series(predict_label, index = X.index)
    
    def score(self, X, y):
       
        y_predict = self.predict(X)
        return (y_predict == y).sum() / y.shape[0]







