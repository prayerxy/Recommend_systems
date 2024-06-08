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

result_path="./Result/"



# latent factor model
class mySVD:
    def __init__(self,items,users,item_user_matrix,lamda1=0.05,lamda2=0.05,lamda3=0.05,lamda4=0.05,factors=50,lr=0.005):
        #测试时需要进行items和users的映射
        self.items=items
        self.users=users
        self.item_user_matrix=item_user_matrix
        
        # 划分训练集和验证集
        self.train_data, self.valid_data = self.split_valid(ratio = 0.8)
        
        self.n_items=len(items)
        self.n_users=len(users)
        #隐向量的维度
        self.factors=factors
        self.lr=lr
        self.lamda1=lamda1
        self.lamda2=lamda2
        self.lamda3=lamda3
        self.lamda4=lamda4
        self.n_epochs=50
        #全局平均分
        self.globle_mean=get_mean()
        #参数初始化
        self.bx=get_bx()
        self.bi=get_bi()
        #items*factor
        self.Q=np.random.normal(0,0.1,(self.n_items,self.factors)).astype(np.float64)
        #users*factor
        self.P=np.random.normal(0,0.1,(self.n_users,self.factors)).astype(np.float64)
        self.history={'train_loss':[],'valid_loss':[]}

    
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

    def loss(self, isValid = False):
        #don't care about the values on missing ones
        loss=0.0
        count=0
        data = self.valid_data if isValid else self.train_data
        for item,users in data.items():
            for user,rating in users:
                count+=1
                loss+=(rating - self.predict(user, item))**2
        #正则项 4个
        if not isValid:
            loss+=self.lamda1*np.sum(self.bx**2)
            loss+=self.lamda2*np.sum(self.bi**2)
            loss+=self.lamda3*np.sum(self.Q**2)
            loss+=self.lamda4*np.sum(self.P**2)
            # loss+=0.5*self.lamda1*np.sum(self.bx**2)
            # loss+=0.5*self.lamda2*np.sum(self.bi**2)
            # loss+=0.5*self.lamda3*np.sum(self.Q**2)
            # loss+=0.5*self.lamda4*np.sum(self.P**2)
        loss/=count
        return loss
    
    # 对验证集计算RMSE
    def caculate_RMSE(self):
        RMSE = 0
        sum_err = 0
        n = 0
        for item, users in self.train_data.items():
            for user, rating in users:
                n+=1
                sum_err += (rating-self.globle_mean-self.bx[user]-self.bi[item]-np.dot(self.Q[item],self.P[user]))**2
        RMSE = np.sqrt(sum_err/n)
        return RMSE
                
    
    def caculate_accuracy(self):
        right_n = 0
        n = 0
        # +- 0.1在接受范围之内
        for item, users in self.train_data.items():
            for user, rating in users:
                predict_score = self.globle_mean+self.bx[user]-self.bi[item]+np.dot(self.Q[item],self.P[user])
                if predict_score >= rating - 0.1 and predict_score <=rating + 0.1:
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
                        grad_bx=-2*error+2*self.lamda1*self.bx[user]
                        grad_bi=-2*error+2*self.lamda2*self.bi[item]
                        grad_Q=-2*error*self.P[user]+2*self.lamda3*self.Q[item]
                        grad_P=-2*error*self.Q[item]+2*self.lamda4*self.P[user]
                        # 更新参数
                        self.bx[user]-=self.lr*grad_bx
                        self.bi[item]-=self.lr*grad_bi
                        self.Q[item]-=self.lr*grad_Q
                        self.P[user]-=self.lr*grad_P
                
                loss=self.loss()
                print('epoch:',epoch+1,'train_loss:',loss)
                valid_loss = self.loss(isValid=True)
                print('epoch:',epoch+1,'valid_loss:',valid_loss)
                
                
                self.history['train_loss'].append(loss)
                self.history['valid_loss'].append(valid_loss)
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
        for loss in self.history['train_loss']:
            print('epoch',epoch_count+1,' loss: ',loss)
            epoch_count+=1
        
        # 对验证集进行计算RMSE
        RMSE = self.caculate_RMSE()
        print('in train_data RMSE is : ', RMSE)
        
        # 计算准确率
        accuracy = self.caculate_accuracy()
        print('accuracy is : ', accuracy)
        

    def predict(self,user,item):
        return self.globle_mean+self.bx[user]+self.bi[item]+np.dot(self.Q[item],self.P[user])
    def show_loss(self):
        train_loss = self.history['train_loss']
        valid_loss = self.history['valid_loss']
        plt.plot(range(self.n_epochs),train_loss)
        plt.plot(range(self.n_epochs),valid_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()
        #保存loss至SVD/loss.txt
        with open('./SVD/loss.txt','w') as f:
            for i in range(len(train_loss)):
                f.write('epoch: '+str(i+1)+' train_loss: '+str(train_loss[i])+' valid_loss: '+str(valid_loss[i])+'\n')

    def test(self):
        test_matrix=load_pickle(pickle_path+'test_matrix.pkl')
        test_result=defaultdict(list)
        for user,items in test_matrix.items():
            for item in items:
                user_index=self.users[user]
                try:
                    item_index=self.items[item]
                    rate=self.predict(user_index,item_index)*10
                    # 注意界限
                    if rate<0.0:
                        rate = 0.0
                    if rate>100.0:
                        rate = 100.0
                    # print("user:",user," item:",item," rate:",rate)
                    test_result[user].append([item,rate])
                except:
                    #使用默认值
                    rate=(self.globle_mean + self.bx[user])*10
                    if rate<0.0:
                        rate = 0.0
                    if rate > 100.0:
                        rate = 100.0
                    # print("user:",user," item:",item," rate(mean):",rate)
                    test_result[user].append([item,rate])
        
        with open(result_path+'test_result.txt','w') as f:
            for user,items in test_result.items():
                f.write(str(user)+'|'+str(len(items))+'\n')
                for item,rate in items:
                    f.write(str(item)+' '+str(rate)+'\n')

if __name__ == '__main__':
    svd=mySVD(items,users,item_user_matrix)
    svd.train(mode='gd')
    svd.test()
    svd.show_loss()
    svd.save_params()
    print('Done')

        

