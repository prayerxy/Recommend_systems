from collections import defaultdict
import random
import numpy as np
from utils import load_pickle,save_pickle,get_bi,get_bx,get_mean,adam
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
        self.item_user_matrix=item_user_matrix
        self.user_item_matrix=user_item_matrix
        self.tmp_item_user_matrix = item_user_matrix
        # consinemap
        self.consinemap = {}
        # 填充未知项
        self.fill_missing()
        
        
        # 划分训练集和验证集
        self.train_data, self.valid_data = self.split_valid(ratio = 0.9)
        
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

    
    def cal_cosin(self, item, sim_item):
        if item_attr1 ==0 and item_attr2 ==0:
            return 0

        if sim_item_attr1 ==0 and sim_item_attr2 ==0:
            return 0
        item_attr1 = item_attrs[item][0]
        item_attr2 = item_attrs[item][1]
        sim_item_attr1 = item_attrs[sim_item][0]
        sim_item_attr2 = item_attrs[sim_item][1]
        res = item_attr1 * sim_item_attr1 + item_attr2 * sim_item_attr2
        res /= np.sqrt((item_attr1**2+item_attr2**2)*(sim_item_attr1**2+sim_item_attr2**2))
        return res
    
    def fill_missing(self):
        thold = 0.8
        for item, users in self.tmp_item_user_matrix.items():
            item_user_list = [sublist[0] for sublist in users]
            if item not in self.items:
                continue
            for user in range(self.n_users):
                if user in item_user_list:
                    continue
                fill_score_son = 0
                fill_score_mom = 0
                fill_score = 0
                for [sim_item, sim_item_rating] in self.user_item_matrix[user]:
                    if sim_item not in self.items:
                        continue
                    # 计算consine相似度
                    cosin = self.cal_cosin(item, sim_item)
                    
                    if cosin >= thold:
                        fill_score_son += cosin*sim_item_rating
                        fill_score_mom += cosin
                if fill_score_mom != 0:
                    fill_score = float(fill_score_son/fill_score_mom)
                self.item_user_matrix[item].append([user, fill_score])
        
        print('over')
            
                
                    
    def split_valid(self, ratio):
        # item_user_matrix
        train_data = defaultdict(list)
        valid_data = defaultdict(list)
        for item, users in self.item_user_matrix.items():
            for user, rating in users:
                #选择一个0-1之间的随机数
                if np.random.rand() < ratio:
                    train_data[item].append([user, rating])
                else:
                    valid_data[item].append([user, rating])
        return train_data, valid_data
        
        
    
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
    
                
    
    def caculate_accuracy(self):
        right_n = 0
        n = 0
        # +- 0.1在接受范围之内
        for item, users in self.train_data.items():
            for user, rating in users:
                predict_score = self.globle_mean+self.bx[user]-self.bi[item]+np.dot(self.Q[item],self.P[user])
                if predict_score > 100:
                    predict_score = 100
                if predict_score <0:
                    predict_score = 0
                if predict_score >= rating - 5 and predict_score <=rating + 5:
                    right_n+=1
                n+=1
        return float(right_n/n)

                
             
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
                
                # 学习率更新
                self.lr *= 0.5
                #每个epoch保存一次参数
                self.save_params()
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
                        grad_bx=-2*error+self.lamda1*self.bx[user]
                        grad_bi=-2*error+self.lamda2*self.bi[item]
                        grad_Q=-2*error*self.P[user]+self.lamda3*self.Q[item]
                        grad_P=-2*error*self.Q[item]+self.lamda4*self.P[user]
                        # 更新参数
                        self.bx[user]-=self.lr*grad_bx
                        self.bi[item]-=self.lr*grad_bi
                        self.Q[item]-=self.lr*grad_Q
                        self.P[user]-=self.lr*grad_P
                
                loss=self.loss()
                print('epoch:', epoch+1, 'train_loss:', loss)
                valid_loss = self.loss(isValid=True)
                print('epoch:', epoch+1, ' valid_loss:', valid_loss)
                self.history['train_loss'].append(loss)
                self.history['valid_loss'].append(valid_loss)
                
                # 每个epoch保存一次参数
                self.save_params()
        
        # 打印所有轮的train_train_loss
        epoch_count = 0
        for rmse in self.history['train_rmse']:
            print('epoch',epoch_count+1,' train_rmse: ',rmse)
            epoch_count+=1
        
        
        # 计算准确率
        accuracy = self.caculate_accuracy()
        print('accuracy is : ', accuracy)
        

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

    def test(self):
        test_matrix=load_pickle(pickle_path+'test_matrix.pkl')
        test_result=defaultdict(list)
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
                    #使用默认值
                    rate=self.globle_mean + self.bx[user]
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
    svd.load_params()
    svd.test()
    svd.show_rmse()
    svd.save_params()
    print('Done')

        

