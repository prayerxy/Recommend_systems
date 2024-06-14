from collections import defaultdict
import random
import numpy as np
from utils import load_pickle,save_pickle,get_bi,get_bx,get_mean,load_true_item
#进度条
from tqdm import tqdm
import matplotlib.pyplot as plt

pickle_path="./mypickle/"

users=load_pickle(pickle_path+'users.pkl')
items=load_pickle(pickle_path+'items.pkl')
item_user_matrix=load_pickle(pickle_path+'item_user_matrix.pkl')
user_item_matrix=load_pickle(pickle_path+'user_item_matrix.pkl')



item_attrs = load_pickle(pickle_path+'item_attrs.pkl')
result_path="./Result/"



# latent factor model
class mySVD:
    def __init__(self,items,users,item_user_matrix,lamda1=0.01,lamda2=0.01,lamda3=0.007,lamda4=0.007,factors=100,lr=0.002):
        #测试时需要进行items和users的映射
        self.items=items
        self.users=users
        self.trueItems = load_true_item()
        self.item_user_matrix=item_user_matrix
        
        # 获取item_attrs
        self.item_attr = item_attrs
        
        # 划分训练集和验证集
        self.train_data, self.valid_data, self.train_user = self.split_valid(ratio = 0.9)
        
        # consinemap
        # 真实id对应
        self.consinemap = defaultdict(dict)
        self.n_items=len(items)
        self.n_users=len(users)
        #隐向量的维度
        self.factors=factors
        self.lr=lr
        self.lamda1=lamda1
        self.lamda2=lamda2
        self.lamda3=lamda3
        self.lamda4=lamda4
        self.n_epochs=20
        #全局平均分
        self.globle_mean=get_mean()
        #参数初始化
        self.bx=get_bx()
        self.bi=get_bi()
        #items*factor
        self.Q=np.random.normal(0,0.1,(self.n_items,self.factors)).astype(np.float64)
        #users*factor
        self.P=np.random.normal(0,0.1,(self.n_users,self.factors)).astype(np.float64)
        self.history={'train_rmse':[],'valid_rmse':[]}
        
        # 最小的vaild_rmse
        self.min_valid_rmse = 1000

    
    def split_valid(self, ratio):
        # item_user_matrix
        train_data = defaultdict(list)
        train_user = defaultdict(dict)
        valid_data = defaultdict(list)
        for item, users in self.item_user_matrix.items():
            for user, rating in users:
                #选择一个0-1之间的随机数
                if np.random.rand() < ratio:
                    train_data[item].append([user, rating])
                    if user not in train_user:
                        train_user[user] = {}
                    train_user[user][item] = rating
                else:
                    valid_data[item].append([user, rating])
        return train_data, valid_data, train_user
        
        
    
    def load_params(self):
        svd_path="./SVD/"
        #检测是否存在
        try:
            self.bx=load_pickle(svd_path+'bx.pkl')
            self.bi=load_pickle(svd_path+'bi.pkl')
            self.Q=load_pickle(svd_path+'Q.pkl')
            self.P=load_pickle(svd_path+'P.pkl')
        except:
            print('No params found')
    def save_params(self):
        #使用pickle
        svd_path="./SVD/"
        save_pickle(self.bx,svd_path+'bx.pkl')
        save_pickle(self.bi,svd_path+'bi.pkl')
        save_pickle(self.Q,svd_path+'Q.pkl')
        save_pickle(self.P,svd_path+'P.pkl')

    def cal_error(self,item,user,rating):
        #使用float记录
        #np.dot进行向量的点积
        error=rating-self.globle_mean-self.bx[user]-self.bi[item]-np.dot(self.Q[item],self.P[user])
        return error

    def cal_RMSE(self, isValid = False):
        #don't care about the values on missing ones
        rmse=0.0
        count=0
        data = self.valid_data if isValid else self.train_data
        for item,users in data.items():
            for user,rating in users:
                count+=1
                rmse+=(rating - self.predict(user, item))**2
        #正则项 4个
        if not isValid:
            rmse+=self.lamda1*np.sum(self.bx**2)
            rmse+=self.lamda2*np.sum(self.bi**2)
            rmse+=self.lamda3*np.sum(self.Q**2)
            rmse+=self.lamda4*np.sum(self.P**2)
            # loss+=0.5*self.lamda1*np.sum(self.bx**2)
            # loss+=0.5*self.lamda2*np.sum(self.bi**2)
            # loss+=0.5*self.lamda3*np.sum(self.Q**2)
            # loss+=0.5*self.lamda4*np.sum(self.P**2)
        rmse/=count
        rmse = np.sqrt(rmse)
        return rmse
    
                
                
             
    def train(self,mode='sgd'):
        #gd方式 全局梯度下降
        if mode=='gd':
            for epoch in range(self.n_epochs):
                for item, users in tqdm(self.train_data.items(), desc=f"Progress [{epoch+1}/{self.n_epochs}]", total=len(self.train_data)):
                    for user, rating in users:
                        error=self.cal_error(item,user,rating)
                        # 计算梯度
                        grad_bx=-error+self.lamda1*self.bx[user]
                        grad_bi=-error+self.lamda2*self.bi[item]
                        grad_Q=-error*self.P[user]+self.lamda3*self.Q[item]
                        grad_P=-error*self.Q[item]+self.lamda4*self.P[user]
                        
                        # 更新参数
                        self.bx[user]-=self.lr*grad_bx
                        self.bi[item]-=self.lr*grad_bi
                        self.Q[item]-=self.lr*grad_Q
                        self.P[user]-=self.lr*grad_P
                
                train_rmse = self.cal_RMSE()
                print('epoch:',epoch+1,'train_rmse:',train_rmse)
                valid_rmse = self.cal_RMSE(isValid=True)
                print('epoch:',epoch+1,'valid_rmse:',valid_rmse)
                
                
                self.history['train_rmse'].append(train_rmse)
                self.history['valid_rmse'].append(valid_rmse)
                
                if valid_rmse < self.min_valid_rmse:
                    self.min_valid_rmse = valid_rmse
                    # 保存最好的模型
                    self.save_params()
                
                # 学习率更新
                self.lr *= 0.5
                
        else:
            #sgd方式 随机梯度下降
            batch_size=1024
            data_list=list(self.train_data.items())
            for epoch in range(self.n_epochs):
                batch=tqdm(random.sample(data_list,batch_size),desc=f"Epoch [{epoch+1}/{self.n_epochs}]",unit="item")
                for item, users in batch:
                    for user, rating in users:
                        error=self.cal_error(item,user,rating)
                        #计算梯度
                        grad_bx=-error+self.lamda1*self.bx[user]
                        grad_bi=-error+self.lamda2*self.bi[item]
                        grad_Q=-error*self.P[user]+self.lamda3*self.Q[item]
                        grad_P=-error*self.Q[item]+self.lamda4*self.P[user]
                        # 更新参数
                        self.bx[user]-=self.lr*grad_bx
                        self.bi[item]-=self.lr*grad_bi
                        self.Q[item]-=self.lr*grad_Q
                        self.P[user]-=self.lr*grad_P
                
                train_rmse = self.cal_RMSE()
                print('epoch:',epoch+1,'train_rmse:',train_rmse)
                valid_rmse = self.cal_RMSE(isValid=True)
                print('epoch:',epoch+1,'valid_rmse:',valid_rmse)
                
                
                self.history['train_rmse'].append(train_rmse)
                self.history['valid_rmse'].append(valid_rmse)
                
                if valid_rmse < self.min_valid_rmse:
                    self.min_valid_rmse = valid_rmse
                    # 保存最好的模型
                    self.save_params()
        
        epoch_count = 0
        for rmse in self.history['train_rmse']:
            print('epoch',epoch_count+1,' train_rmse: ',rmse)
            epoch_count+=1
        
        
        

    def predict(self,user,item):
        return self.globle_mean+self.bx[user]+self.bi[item]+np.dot(self.Q[item],self.P[user])
    def show_rmse(self):
        train_rmse = self.history['train_rmse']
        valid_rmse = self.history['valid_rmse']
        plt.plot(range(self.n_epochs),train_rmse)
        plt.plot(range(self.n_epochs),valid_rmse)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Epoch')
        plt.show()
        #保存rmse至SVD/rmse.txt
        with open('./SVD/rmse.txt','w') as f:
            for i in range(len(train_rmse)):
                f.write('epoch: '+str(i+1)+' train_rmse: '+str(train_rmse[i])+' valid_loss: '+str(valid_rmse[i])+'\n')

    
    def get_true_item(self, item):
        return self.trueItems[item]
    
    
    def get_similarity(self, item_i, item_j):
        consine_sim = None
        if item_i in self.consinemap and item_j in self.consinemap[item_i]:
            consine_sim = self.consinemap[item_i][item_j]
        elif item_j in self.consinemap and item_i in self.consinemap[item_j]:
            consine_sim = self.consinemap[item_j][item_i]
        else:
            consine_sim = None
        return consine_sim
    
    
    # 传入映射user_id，真实的item
    def calc_similar_item(self, user, item_i):
        similar_item = {}
        for item_j in self.train_user[user].keys():
            true_item_j = self.get_true_item(item_j)
            similar_res = self.get_similarity(item_i, true_item_j)
            
            if similar_res is None:
                if self.item_attr[item_i][2] == 0 or self.item_attr[true_item_j][2]==0:
                    similar_res = 0
                else:
                    similar_res = (self.item_attr[item_i][0]*self.item_attr[true_item_j][0]
                           + self.item_attr[item_i][1]*self.item_attr[true_item_j][1])/(self.item_attr[item_i][2]*self.item_attr[true_item_j][2])
                if similar_res!=0:
                    if item_i not in self.consinemap:
                        self.consinemap[item_i] = {}
                    self.consinemap[item_i][true_item_j] = similar_res
                    
            # 设置阈值为0.9
            if similar_res >=0.9:
                similar_item[true_item_j] = similar_res
            
        return similar_item
    
    
    
    def test(self):
        test_matrix=load_pickle(pickle_path+'test_matrix.pkl')
        test_result=defaultdict(list)
        self.load_params()
        for user,items in test_matrix.items():
            for item in items:
                user_index=self.users[user]
                try:
                    item_index=self.items[item]
                    rate=self.predict(user_index,item_index)
                    # 注意界限
                    if rate<0.0:
                        rate = 0.0
                    if rate>100.0:
                        rate = 100.0
                    # print("user:",user," item:",item," rate:",rate)
                    test_result[user].append([item,rate])
                except:
                    #引入属性相似度
                    rate = 0
                    bias_i=self.globle_mean + self.bx[user]
                    if item in self.item_attr:
                        # 里面存的是真实ID
                        similar_item = self.calc_similar_item(user_index, item)
                        similar_item = sorted(similar_item.items(), key = lambda item: item[1], reverse = True)
                        norm = 0
                        for i, (item_j, similarity) in enumerate(similar_item):
                            if i > 200:
                                break
                            item_j_index = self.items[item_j]
                            bias_j = self.globle_mean + self.bx[user_index] + self.bi[item_j_index]
                            rate+=similarity*(self.train_user[user_index][item_j_index] - bias_j)
                            norm+=similarity
                        if norm==0:
                            rate = 0
                        else:
                            rate /= norm
                    rate+=bias_i
                    if rate<0.0:
                        rate = 0.0
                    if rate > 100.0:
                        rate = 100.0
                    # print("user:",user," item:",item," rate(mean):",rate)
                    test_result[user].append([item,rate])
        
        with open(result_path+'result_svd.txt','w') as f:
            for user,items in test_result.items():
                f.write(str(user)+'|'+str(len(items))+'\n')
                for item,rate in items:
                    f.write(str(item)+' '+str(rate)+'\n')

if __name__ == '__main__':
    svd=mySVD(items,users,item_user_matrix)
    svd.train(mode='gd')
    # svd.load_params()
    svd.test()
    svd.show_rmse()
    # svd.save_params()
    print('Done')

        

