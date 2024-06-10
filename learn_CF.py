from collections import defaultdict
from utils import load_pickle, save_pickle, get_mean
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# user_item_data和item_user_data路径
user_item_pkl = './mypickle/user_item_matrix.pkl'
item_user_pkl = './mypickle/item_user_matrix.pkl'

# 存储bias路径
bx_path = './mypickle/default_bx.pkl'
bi_path = './mypickle/default_bi.pkl'

# test数据pickle路径
test_data_path = './mypickle/test_matrix.pkl'

# user_idx item_idx
users_idx = './mypickle/users.pkl'
items_idx = './mypickle/items.pkl'

# 写入结果路径
test_predict_result_path = './Result/result_learn_CF.txt'

# item和user数量
items_num = 455691
users_num = 19835

class my_learnCF:
    def __init__(self, lr):
        self.bx = load_pickle(bx_path)
        self.bi = load_pickle(bi_path)
        self.miu = get_mean()
        self.user_idx = load_pickle(users_idx)
        self.item_idx = load_pickle(items_idx)
        self.user_item_data = load_pickle(user_item_pkl)
        self.item_user_data = load_pickle(item_user_pkl)
        self.test_data = load_pickle(test_data_path)
        
        # 划分训练集和验证集
        self.train_train_data , self.train_test_data = self.split_item_user_matrix(ratio = 0.9)
        
        # 训练轮数
        self.n_epochs = 50
        # 随机初始化w
        self.w = np.ones((items_num, items_num))
        self.lr = lr
        self.history={'loss':[]}
    
    def split_item_user_matrix(self, ratio):
        # 依照item_user_matrix来划分训练集train_train_data和train_test_data
        train_train_data = defaultdict(list)
        train_test_data = defaultdict(list)
        for item, users in self.item_user_data.items():
            for user, rating in users:
                # 选择一个0-1之间的随机数
                if np.random.rand() < ratio:
                    train_train_data[item].append([user, rating])
                else:
                    train_test_data[item].append([user, rating])
        return train_train_data, train_test_data
    
    
    def load_params(self):
        learn_CF_path = "./learn_CF/"
        try:
            self.w = load_pickle(learn_CF_path+'w.pkl')
        except:
            print('No params found')
    
    
    def save_params(self):
        learn_CF_path = "./learn_CF/"
        save_pickle(self.w, learn_CF_path+'w.pkl')
    
    
    
    
    def caculate_bxi(self, user_x, item_i):
        return self.miu + self.bx[user_x] + self.bi[item_i]
    
    def caculate_sum_w(self, i_id, u_id):
        sum_w = 0
        for [k_id, k_score] in self.user_item_data[u_id]:
            sum_w+=self.w[i_id,k_id]*(k_score-self.caculate_bxi(u_id, k_id))
        return sum_w
    
    def loss(self, is_train_test = False):
        loss = 0.0
        count = 0
        data = self.train_test_data if is_train_test else self.train_train_data
        for item, users in data.items():
            for user, rating in users:
                # 开始预测
                count+=1
                loss+=(rating - self.predict(user, item))**2
        
        loss/=count
        return loss
    
    def caculate_RMSE(self):
        RMSE = 0
        sum_err = 0
        n = 0
        for item, users in self.train_test_data.items():
            for user, rating in users:
                n+=1
                sum_err += (rating-self.predict(user, item))**2
        RMSE = np.sqrt(sum_err/n)
        return RMSE
    
    def caculate_accuracy(self):
        right_n = 0
        n = 0
        # +- 0.1在接受范围之内
        for item, users in self.train_test_data.items():
            for user, rating in users:
                predict_score = self.predict(user, item)
                if predict_score >= rating - 0.1 and predict_score <=rating + 0.1:
                    right_n+=1
                n+=1
        return float(right_n/n)
    
    def predict(self,user,item):
        bxi = self.caculate_bxi(user, item)
        sum_w =  self.caculate_sum_w(item, user)
        predict_score = bxi + sum_w
        return predict_score
            
        
    def train(self):
        for epoch in range(self.n_epochs):
            for i_id, users in tqdm(self.train_train_data.items(), desc=f"Progress [{epoch+1}/{self.n_epochs}]", total=len(self.train_train_data)):
                for u_id, i_score in users:
                    # 固定i_id和u_id 遍历所有的j_id
                    sum_w = self.caculate_sum_w(i_id, u_id)
                    for [j_id, j_score] in self.user_item_data[u_id]:
                        # 计算梯度W i_id sim_item
                        grab_wij = 2*(self.caculate_bxi(u_id, i_id)+sum_w-i_score)*(j_score-self.caculate_bxi(u_id, j_id))
                        self.w[i_id, j_id]-=self.lr*grab_wij
            loss=self.loss()
            print('epoch:',epoch,' train_train loss:',loss)
            test_loss = self.loss(is_train_test=True)
            print('epoch:',epoch,' train_test loss:',test_loss)
            self.history['loss'].append(loss)
            
            # 每个epoch保存一次参数
            self.save_params()
            
        # 对验证集进行计算RMSE
        RMSE = self.caculate_RMSE()
        print('in train_test_data RMSE is : ', RMSE)
        
        # 计算准确率
        accuracy = self.caculate_accuracy()
        print('accuracy is : ', accuracy)
        

    def show_loss(self):
        losses=self.history['loss']
        plt.plot(range(self.n_epochs),losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()
        
    def test_write(self):
        test_result=defaultdict(list)
        for user, items in self.test_data.items():
            for item in items:
                user_index = self.user_idx[user]
                if item in self.item_idx:
                    item_index = self.item_idx[item]
                    predict_score = self.predict(user_index, item_index)
                    predict_score*=10
                    if predict_score<0:
                        predict_score = 0
                    if predict_score>100:
                        predict_score = 100
                    test_result[user].append([item, predict_score])
                else:
                    predict_score = self.miu + self.bx[self.user_idx[user]]
                    predict_score*=10
                    if predict_score<0:
                        predict_score = 0
                    if predict_score>100:
                        predict_score = 100
                    test_result[user].append([item, predict_score])
        
        # 写入指定路径
        with open(test_predict_result_path, 'w') as f:
            for user, items in test_result.items():
                f.write(str(user)+'|'+str(len(items))+'\n')
                for item,rate in items:
                    f.write(str(item)+' '+str(rate)+'\n')
                
                    
        
if __name__ == '__main__':
    svd=my_learnCF(lr=5e-3)
    svd.train()
    svd.test_write()
    svd.show_loss()
    svd.save_params()
    print('Done')     
            
            
        
                    
        
        