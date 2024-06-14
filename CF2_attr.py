from collections import defaultdict
import random

from tqdm import tqdm
from utils import load_pickle, save_pickle, get_mean,load_true_item
import numpy as np

# user_item_data和item_user_data路径
user_item_pkl = './mypickle/user_item_matrix.pkl'
item_user_pkl = './mypickle/item_user_matrix.pkl'

# 存储bias路径
bx_path = './mypickle/default_bx.pkl'
bi_path = './mypickle/default_bi.pkl'
rmse_path='./Result/cf_rmse.txt'
# test数据pickle路径
test_data_path = './mypickle/test_matrix.pkl'

pickle_path="./mypickle/"

# user_idx item_idx
users_idx = './mypickle/users.pkl'
items_idx = './mypickle/items.pkl'

# item -> [attr1, attr2, norm]
item_attrs = load_pickle(pickle_path+'item_attrs.pkl')

# 写入结果路径
test_predict_result_path = './Result/result_CF_attr.txt'
# item和user数量
items_num = 455691
users_num = 19835

class my_basicCF:
    def __init__(self):
        self.trueItems=load_true_item()
        self.bx = load_pickle(bx_path)
        self.bi = load_pickle(bi_path)
        self.miu = get_mean()
        self.user_idx = load_pickle(users_idx)
        self.item_idx = load_pickle(items_idx)
        # self.user_item_data = load_pickle(user_item_pkl)
        self.item_user_data = load_pickle(item_user_pkl)
        
        # 获取item_attrs
        self.item_attr = item_attrs
        
        # 相似集
        # 映射id
        self.simmap = {}
        
        # consine_map
        self.consinemap = defaultdict(dict)
        

        # 划分验证集
        self.train_item_data, self.train_user_data,self.valid_item_data,self.train_user, self.valid_num= self.split_valid(ratio = 0.95)
        self.test_data = load_pickle(test_data_path)


    def split_valid(self, ratio):
        # item_user_matrix
        train_item_data = defaultdict(dict)
        train_user = defaultdict(dict)
        valid_item_data = defaultdict(list)
        train_user_data = defaultdict(list)
        count=0
        k=0
        for item, users in self.item_user_data.items():
            for user, rating in users:
                #选择一个0-1之间的随机数
                np.random.seed(k)
                k+=1
                if np.random.rand() < ratio:
                    train_item_data[item][user] = rating
                    train_user_data[user].append([item, rating])
                    if user not in train_user:
                        train_user[user] = {}
                    train_user[user][item] = rating

                else:
                    valid_item_data[item].append([user, rating])
                    count+=1
        return train_item_data, train_user_data,valid_item_data,train_user, count



    def caculate_bxi(self, user_x, item_i):
        return self.miu + self.bx[user_x] + self.bi[item_i]


    def caculate_person_sim(self, item1, item2):
        sim = 0
        # 计算item1与item2的相似度
        # 使用train_user_data train_item_data
        user_scores1 = self.train_item_data[item1]
        user_scores2 = self.train_item_data[item2]

        # 获取用户ID列表
        users1 = set(user_scores1.keys())
        users2 = set(user_scores2.keys())

        # 计算用户ID的交集
        common_users = users1.intersection(users2)

        # 计算分子
        son = 0
        # 计算分母
        mother = 0

        miu1 = self.bi[item1] + self.miu
        miu2 = self.bi[item2] + self.miu

        for user in common_users:
            score1 = user_scores1[user]
            score2 = user_scores2[user]
            son += (score1 - miu1) * (score2 - miu2)
            mother += (score1 - miu1) ** 2 * (score2 - miu2) ** 2

        if mother != 0:
            sim = son / np.sqrt(mother)

        return sim
    
    def get_true_item(self, item):
        return self.trueItems[item]
    
    # 传入的是真实id
    def caculate_consine(self,i_id,item):
        res = 0
        if self.item_attr[i_id][2] == 0 or self.item_attr[item][2] == 0:
            res = 0
        else :
            res = (self.item_attr[i_id][0]*self.item_attr[item][0]
                           + self.item_attr[i_id][1]*self.item_attr[item][1])/(self.item_attr[i_id][2]*self.item_attr[item][2])
        return res
    
    
    def train(self):
        # 采取分块计算策略
        sum_RMSE = 0
        print('begin train')
        data=list(self.valid_item_data.items())[39008:48760]
        num=0
        for i_id, i_id_ratings in tqdm(data,desc='valid_item_data',total=len(data)):
            # print('i_id ', i_id, ' begin caculate')
            for u_id, i_score in i_id_ratings:
                # 开始预测u_id给i_id打分
                
                # 临时存储u打分了的item与这个item的相似度
                # item -> sim
                sim_item_dict = {}
                
                # 存储u_id给sim_item打的分
                u_sim_score = {}

                # 预测打分 以baseline作下界 son和mother代表分子与分母
                predict_score = self.miu + self.bx[u_id] + self.bi[i_id]
                predict_score_son = 0
                predict_score_mother = 0
                #并行下面的循环


                for item,rating in self.train_user_data[u_id]:
                    # print(len(self.train_user_data[u_id]))
                    #和训练集的每个Item计算相似度
                    # 计算相似度
                    if (i_id, item) in self.simmap or (item, i_id) in self.simmap:
                        sim_res = self.simmap[(i_id, item)] if (i_id, item) in self.simmap else self.simmap[(item, i_id)]
                    else:
                        sim_res = self.caculate_person_sim(i_id, item)
                        true_i_id = self.get_true_item(i_id)
                        true_item = self.get_true_item(item)
                        consine_res = self.caculate_consine(true_i_id, true_item)
                        sim_res = (sim_res + consine_res)/2
                        self.simmap[(i_id, item)] = sim_res
                    if sim_res!=0:
                        sim_item_dict[item] = sim_res
                        u_sim_score[item] = rating



                # 计算的相似度进行排序
                sim_item_dict = sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True)
                count = 0
                for (item, person_sim) in sim_item_dict:
                    predict_score_son += person_sim * (u_sim_score[item] -self.miu-self.bx[u_id]-self.bi[item])
                    predict_score_mother += person_sim
                    count+=1
                    if count == 400:
                        break
                if predict_score_mother!=0:
                    predict_score += predict_score_son / predict_score_mother
                    predict_score = min(100.0, max(0.0, predict_score))
                num+=1
                sum_RMSE += ((predict_score - i_score)**2)

        #保存simmap
        # self.save_params()
        sum_RMSE=np.sqrt(sum_RMSE/num)
        print('RMSE: ', sum_RMSE)
        print('num: ', num)
        with open(rmse_path,'w') as f:
            f.write(str(sum_RMSE)+' '+str(num)+'\n')
        return sum_RMSE
    
    def save_params(self):
        #使用pickle
        CF_path="./CF/"
        save_pickle(self.simmap, CF_path+'simmap.pkl')
    
    # 从map中获取consine相似度
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
            
            # 设置相似度阈值为0.9
            if similar_res >=0.9:
                similar_item[true_item_j] = similar_res
            
        return similar_item
    
    
    def test_write(self):
        #模仿valid_test过程
        test_result=defaultdict(list)
        for u_id, i_id_list in tqdm(self.test_data.items(), desc=f"Progress ", total=len(self.test_data)):
            u_true_id = u_id
            u_id = self.user_idx[u_id]
            for i_id in i_id_list:
                i_true_id = i_id
                if i_id not in self.item_idx:
                    rate = 0
                    bias_i=self.miu+self.bx[u_id]
                    similar_item = self.calc_similar_item(u_id, i_true_id)
                    similar_item = sorted(similar_item.items(), key = lambda item: item[1], reverse = True)
                    norm = 0
                    for i, (item_j, similarity) in enumerate(similar_item):
                        if i > 200:
                            break
                        item_j_index = self.item_idx[item_j]
                        bias_j = self.miu + self.bx[u_id] + self.bi[item_j_index]
                        rate+=similarity*(self.train_user[u_id][item_j_index] - bias_j)
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
                    test_result[u_true_id].append([i_true_id, rate])
                else:
                    
                    i_id = self.item_idx[i_id]
                    # 临时存储u打分了的item与这个item的相似度
                    # item -> sim
                    sim_item_dict = {}
                    
                    # 存储u_id给sim_item打的分
                    u_sim_score = {}

                    # 预测打分 以baseline作下界 son和mother代表分子与分母
                    predict_score = self.miu + self.bx[u_id] + self.bi[i_id]
                    predict_score_son = 0
                    predict_score_mother = 0
                    for item,rating in self.train_user_data[u_id]:
                        #和训练集的每个Item计算相似度
                            # 计算相似度
                        if (i_id, item) in self.simmap or (item, i_id) in self.simmap:
                            sim_res = self.simmap[(i_id, item)] if (i_id, item) in self.simmap else self.simmap[(item, i_id)]
                        else:
                            sim_res = self.caculate_person_sim(i_id, item)
                            true_item = self.get_true_item(item)
                            # 传入真实id
                            consine_res = self.caculate_consine(i_true_id, true_item)
                            sim_res = (sim_res + consine_res)/2
                            self.simmap[(i_id, item)] = sim_res
                        if sim_res!=0:
                            sim_item_dict[item] = sim_res
                            u_sim_score[item] = rating

                    # 计算的相似度进行排序
                    sim_item_dict = sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True)
                    count = 0
                    for (item, person_sim) in sim_item_dict:
                        predict_score_son += person_sim * (u_sim_score[item] -self.miu-self.bx[u_id]-self.bi[item])
                        predict_score_mother += person_sim
                        count+=1
                        if count == 400:
                            break
                    if predict_score_mother!=0:
                        predict_score += predict_score_son / predict_score_mother
                    predict_score = min(100.0, max(0.0, predict_score))
                    test_result[u_true_id].append([i_true_id, predict_score])
        
        # 写入指定路径
        with open(test_predict_result_path, 'w') as f:
            for u_id, item_ratings in test_result.items():
                num = len(item_ratings)
                f.write(str(u_id)+'|'+str(num)+'\n')
                for i_id, i_rating in item_ratings:
                    f.write(str(i_id)+' '+str(i_rating)+'\n')
                    
                    
if __name__ == '__main__':
    mycf = my_basicCF()
    # train数据集上计算RMSE
    # RMSE = mycf.train()
    print('begin CF_attr test')
    mycf.test_write()
    
    # print('RMSE: ', RMSE)
    print('over')
        
                    