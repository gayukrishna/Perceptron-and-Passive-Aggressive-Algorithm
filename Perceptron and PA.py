# Created by: Gayathri Krishnamoorthy
# Updated: 03-23-2020

# Binary and Multi-Class classifier using Perceptron and Passive Aggressive algorithms for fashion Mnist data is implemented here.
# I have BinaryTrain, BinaryTest, MulticlassTrain and MulticlassTest in a python class called perceptron. 
# Boolean variables are created in each function definitions to indicate whether Perceptron or Passive Aggressive weight update is used. 
# It is coded in python version 3.6.

import numpy as np
import csv
import os
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

class Perceptron:
    def __init__(self, train_path, test_path):
        try:
            self.x_train = np.load('./np_arrays/x_train.npz')
            self.y_train = np.load('./np_arrays/y_train.npz')
        except:
            self.x_train, self.y_train = self.ReadData(train_path)
            if not(os.path.exists('./np_arrays/')):
                os.mkdir('./np_arrays')
            np.save('./np_arrays/x_train.npy', self.x_train)
            np.save('./np_arrays/y_train.npy', self.y_train)
        try:
            self.x_test = np.load('./np_arrays/x_test.npy')
            self.y_test = np.load('./np_arrays/y_test.npy')
        except:
            self.x_test, self.y_test = self.ReadData(test_path)
            if not(os.path.exists('./np_arrays/')):
                os.mkdir('./np_arrays')
            np.save('./np_arrays/x_test', self.x_test)
            np.save('./np_arrays/y_test', self.y_test)
        self.feat_len = np.size(self.x_train,1)
        self.label_len = np.size(self.y_train,1)
        self.train_set_len = np.size(self.x_train,0)
        self.test_set_len = np.size(self.x_test,0)


    def ReadData(self, path):
        with open(path, encoding='utf-8-sig') as fn:
            csvreader = csv.reader(fn)
            next(csvreader) ### SKIP COLUMN HEADS ###

            label_len = 10
            feat_len = 784
            x = []
            y = []
            for row in csvreader:
                x.append(list(float(row[i]) for i in range(1,feat_len+1)))
                temp_list = [0]*label_len
                temp_list[int(row[0])] = 1
                y.append(temp_list)
            x = np.array(x)
            y = np.array(y)

        return x, y

    def Binarize(self, y):
        _y = np.zeros(np.size(y,0))
        for i in range(np.size(y,0)):
            #print('y:', y[i])
            odds = np.array([y[i,0],y[i,2],y[i,4],y[i,6],y[i,8]])
            #print('y_odds:',odds)
            if np.sum(odds) == 0:
                _y[i] = 1
            else:
                _y[i] = -1
        return _y
    
##############################################################################################################################################
################################################################################################################################
###############################################################################################

# Binary classifier on training data   
    def TrainBinary(self, max_iter,  k=1, PA = False, AvP = False, vary_train_length = False):
        y_train_bin = self.Binarize(self.y_train)

        '''initialize vectors'''
        w = np.zeros(self.feat_len)
        weight = []
        w_20 = np.zeros(self.feat_len)
        w_sum = np.zeros(self.feat_len)  # for averaged perceptron
        w_log = 1                       # for averaged perceptron        
        acc = np.zeros(max_iter)
        mistakes = np.zeros(max_iter)
        correct = np.zeros(max_iter)
        
        '''perceptron algorithm'''
        for t in range(1,max_iter+1):
            print('######    Starting iteration: ', t, '    ######')
            tmp_logger = AccuLogger()
            ''' Whether or not training set is varied'''
            if vary_train_length ==  False:
                '''normal'''
                train_set_length = self.train_set_len
            else:
                '''varying training set length'''
                train_set_length = k
                
            '''Training Algorithm'''    
            for i in range(train_set_length):
                y_hat = np.sign(np.dot(w,self.x_train[i,:]))
                if y_hat != y_train_bin[i]:
                    '''incorrect while training'''
                    tmp_logger.wrong()
                    ''' Averaged Perceptron or not'''
                    if AvP == False:
                        '''Passive aggressive or not'''
                        if PA == False:
                            tau = 1
                        else:
                            tau = (1-y_train_bin[i]*np.dot(w,self.x_train[i,:]))/np.dot(self.x_train[i,:],self.x_train[i,:])
                            
                        w = w + tau * y_train_bin[i] * self.x_train[i,:]  #update weight for perceptron and PA
                    else:
                        # average perceptron implementation
                        tau = 1                        
                        w = w + tau * y_train_bin[i] * self.x_train[i,:] 
                        w_sum = w_sum + y_train_bin[i] * w_log * self.x_train[i,:]
                        
                else:
                    tmp_logger.right()
                    
                if AvP == True:
                    w=w_sum
                    w_log += 1    
              
            mistakes[t-1]=tmp_logger.wrong()
            acc[t-1] = tmp_logger.get_acc()
            correct[t-1]=tmp_logger.right()
            weight.append(w) 
            
            if t == 20:
                w_20 = w  
                
            print('######    Completed iteration: ', t, 'Accuracy:', acc[t-1], 'Mistakes:', mistakes[t-1], 'Correct:', correct[t-1],   ' ######')                
        #if AvP  == True:
        #    w = w_sum  
            
        return w, w_20, weight, mistakes, acc

##############################################################################################################################################
################################################################################################################################
###############################################################################################

# Multi-class classifier on training data    
    def TrainMulti(self, max_iter, k=1, PA = False, AvP = False, vary_train_length = False):
        '''initialize vectors'''
        w = np.zeros(self.feat_len * self.label_len) #7480 length weight vector
        weight = []
        w_sum = np.zeros(self.feat_len * self.label_len)  # for averaged perceptron
        w_log = 1                       # for averaged perceptron          
        acc = np.zeros(max_iter)
        mistakes = np.zeros(max_iter)
        correct = np.zeros(max_iter)
        
        '''perceptron algorithm'''
        for t in range(1,max_iter+1): # For max iterations
            print('######    Starting iteration: ', t, '    ######')
            tmp_logger = AccuLogger() # keeps track of accuracy
            ''' Whether or not training set is varied'''
            if vary_train_length ==  False:
                '''normal'''
                train_set_length = self.train_set_len
            else:
                '''varying training set length'''
                train_set_length = k
                
            '''Training Algorithm'''    
            for i in range(train_set_length): # for each example
                score = -1 # Initialize any negative score
                multi_feat = np.zeros(self.feat_len * self.label_len) # initialize the predicted F(x,y_hat)
                for j in range(self.label_len): # for each class label (greedy search)
                    tmp_y = np.zeros(self.label_len)
                    tmp_y[j] = 1 # creates label to be tested
                    _multi_feat = np.kron(tmp_y, self.x_train[i,:]) # F(x,y)
                    _score = np.dot(w,_multi_feat) # Temporary score to find highest score
                    if _score > score: # Finds maximum dot product -> predicts associated label
                        score = _score
                        y_hat = tmp_y
                        multi_feat = _multi_feat
                #print('y_hat:', y_hat, 'y_star:', self.y_train[i,:])
                if not np.array_equal(y_hat,self.y_train[i,:]): # Check for error
                    true_multi_feat = np.kron(self.y_train[i,:],self.x_train[i,:]) # correct F(x,y)
                    '''incorrect while training'''
                    tmp_logger.wrong()
                    '''Average Perceptron or not'''
                    if AvP == False:
                        '''Passive aggressive or not'''
                        if PA == False:
                            tau = 1
                        else:
                            tau = (1-(np.dot(w,true_multi_feat)-np.dot(w,multi_feat)))/np.dot(true_multi_feat - multi_feat,true_multi_feat - multi_feat)
                            
                        w = w + tau * (true_multi_feat - multi_feat) # update weight for perceptron and PA
                    
                    else:
                        # average perceptron implementation
                        tau = 1
                        w = w + tau * (true_multi_feat - multi_feat)
                        w_sum = w_sum + w_log * w
                        
                else:
                    tmp_logger.right()
                
                if AvP == True:
                    w = w_sum
                    w_log += 1

            acc[t-1] = tmp_logger.get_acc() 
            mistakes[t-1]=tmp_logger.wrong()
            correct[t-1]=tmp_logger.right()
            weight.append(w)  
            if t == 20:
                w_20 = w             
            print('######    Completed iteration: ', t, 'Accuracy:', acc[t-1], 'Mistakes:', mistakes[t-1], 'Correct:', correct[t-1], '    ######')  
            
        return w, w_20, weight, mistakes, acc 
    
##############################################################################################################################################
################################################################################################################################
###############################################################################################

# Binary classifier on testing data    
    def TestBinary(self,w):
        
        y_test_bin = self.Binarize(self.y_test)
        logger = AccuLogger()
        
        for i in range(self.test_set_len):
            y_hat = np.sign(np.dot(w,self.x_test[i,:]))
            if y_hat != y_test_bin[i]:
                logger.wrong()
            else:
                logger.right()                      
        return logger.get_acc()
    
##############################################################################################################################################
################################################################################################################################
###############################################################################################    

# Multi-class classifier on testing data
    def TestMulti(self,w):

        logger = AccuLogger()
                
        for i in range(self.test_set_len):
            score = -1 # Initialize any negative score
            for j in range(self.label_len):
                tmp_y = np.zeros(self.label_len)
                tmp_y[j] = 1 # creates label to be tested
                _multi_feat = np.kron(tmp_y, self.x_test[i,:]) # F(x,y)
                _score = np.dot(w,_multi_feat) # Temporary score to find highest score
                if _score > score: # Finds maximum dot product -> predicts associated label
                    score = _score
                    y_hat = tmp_y
            if not np.array_equal(y_hat,self.y_test[i,:]): # Check for error
                '''incorrect while testing'''
                logger.wrong()
            else:
                logger.right()                        
        return logger.get_acc()       
    

# logging the right and wrong predictions and calculating accuracy
'''lol this dumb'''
class AccuLogger:
    def __init__(self):
        self.wrongs = 0
        self.rights = 0
        self.acc = 0

    def right(self):
        self.rights += 1
        return self.rights

    def wrong(self):
        self.wrongs += 1
        return self.wrongs
    
    def get_acc(self):
        tot = self.rights + self.wrongs
        self.acc = self.rights/tot
        return self.acc


if __name__ == "__main__":
    train_path = './fashionmnist/fashion-mnist_train.csv'
    test_path = './fashionmnist/fashion-mnist_test.csv'
    perceptron = Perceptron(train_path, test_path)


#############################################################################################################################################
#############################################################################################################################################
#######################################################################

# Here, the online learning curve for the binary classifier using perceptron and passive aggressive (PA) weight updates is analyzed for 50 training iterations. 
# The number of mistakes made by the classifier for each training iteration is plotted.

   
binary_w_perceptron, binary_w20_perceptron, binary_weight_perceptron, mistakes_perceptron, accuracy_perceptron  = perceptron.TrainBinary(50, PA = False)  # Perceptron classifier
binary_w_PA, binary_w20_PA, binary_weight_PA, mistakes_PA, accuracy_PA = perceptron.TrainBinary(50, PA = True) # PA classfier

max_iter=50
training_iterations= range(1,max_iter+1)    
plt.plot(training_iterations, mistakes_perceptron, 'bs',  label='Perceptron', linewidth=2.0)
plt.plot(training_iterations, mistakes_PA,  'g^', label='Passive Aggressive', linewidth=2.0)
plt.xlabel('Number of training iterations')
plt.ylabel('Number of Mistakes in each iteration') 
plt.title('Number of mistakes in each iteration by binary classifier') 
plt.legend()
plt.show()      


# Here, the online learning curve for the multi-class classifier using perceptron and passive aggressive (PA) weight updates is analyzed for 50 training iterations.# 
# The number of mistakes made by the classifier for each training iteration is plotted.

multi_w_perceptron, multi_w20_perceptron,  multi_weight_perceptron, mistakes_perceptron, accuracy_perceptron  = perceptron.TrainMulti(50, PA = False) # Perceptron classifier
multi_w_PA, multi_w20_PA, multi_weight_PA, mistakes_PA, accuracy_PA = perceptron.TrainMulti(50, PA = True) # PA classfier

max_iter=50
training_iterations= range(1,max_iter+1)    
plt.plot(training_iterations, mistakes_perceptron, 'bs',  label='Perceptron', linewidth=2.0)
plt.plot(training_iterations, mistakes_PA,  'g^', label='Passive Aggressive', linewidth=2.0)
plt.xlabel('Number of training iterations')
plt.ylabel('Number of Mistakes in each iteration')  
plt.title('Number of mistakes in each iteration by multi-class classifier')
plt.legend()
plt.show()      

