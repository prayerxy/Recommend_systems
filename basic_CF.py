from collections import defaultdict
from utils import load_pickle, save_pickle, get_mean
import numpy as np

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
test_predict_result_path = './Result/result_basic_CF.txt'
# item和user数量
items_num = 455691
users_num = 19835

class my_basicCF:
    def __init__(self):
        self.bx = load_pickle(bx_path)
        self.bi = load_pickle(bi_path)
        self.miu = get_mean()
        self.user_idx = load_pickle(users_idx)
        self.item_idx = load_pickle(items_idx)
        # self.user_item_data = load_pickle(user_item_pkl)
        self.item_user_data = load_pickle(item_user_pkl)
        
        # 相似集
        self.simmap = defaultdict(dict)
        # 划分验证集
        self.train_item_data, self.valid_item_data, self.train_user_data, self.valid_user_data = self.split_valid(ratio = 0.9)
        del self.item_user_data
        self.test_data = load_pickle(test_data_path)


    def split_valid(self, ratio):
        # item_user_matrix
        train_item_data = defaultdict(list)
        valid_item_data = defaultdict(list)
        train_user_data = defaultdict(list)
        valid_user_data = defaultdict(list)
        for item, users in self.item_user_data.items():
            for user, rating in users:
                #选择一个0-1之间的随机数
                if np.random.rand() < ratio:
                    train_item_data[item].append([user, rating])
                    train_user_data[user].append([item, rating])
                else:
                    valid_item_data[item].append([user, rating])
                    valid_user_data[user].append([item, rating])
        return train_item_data, valid_item_data,train_user_data,valid_user_data



    def caculate_bxi(self, user_x, item_i):
        return self.miu + self.bx[user_x] + self.bi[item_i]

    # 传入的是映射的i_id, sim_item 返回sim_item的相似度
    def caculate_person_sim(self, i_id, sim_item):
        avg_item = self.bi[i_id] + self.miu
        avg_sim_item = self.bi[sim_item] + self.miu
        # 计算相似度的分子,分母左右欧拉距离
        sim_son = 0
        sim_dis_left = 0
        sim_dis_right = 0
        # sim_res
        sim_res = 0
        # 找给sim_item和item都打过分的用户
        # 遍历给item打分的用户是否给sim_item打过分
        for [user_i, user_i_rating] in self.train_item_data[i_id]:
            # 给i_id打过分的所有的
            sim_dis_left += (user_i_rating - avg_item)**2
            for [user_s, user_s_rating] in self.train_item_data[sim_item]:
                if user_i == user_s:
                    sim_son += (user_i_rating - avg_item)*(user_s_rating - avg_sim_item)
                # 给sim_item打过分的所有的
                sim_dis_right += (user_s_rating - avg_sim_item)**2
                        
        if sim_dis_left!=0 and sim_dis_right!=0:
            sim_res = sim_son/((sim_dis_left * sim_dis_right)**(1/2))
        if sim_res != 0:
            if i_id not in self.simmap:
                self.simmap[i_id] = {}
            if sim_item not in self.simmap:
                self.simmap[sim_item] = {}
            self.simmap[i_id][sim_item] = sim_res
            self.simmap[sim_item][i_id] = sim_res
        # 否则sim_res仍然为0
        return sim_res
        
        
    
    def train(self):
        # 采取分块计算策略
        sum_RMSE = 0
        # 统计打分个数
        sum_n = 0
        # 直接对验证集下手
        print('begin train')
        for i_id, i_id_ratings in self.valid_item_data.items():
            print('i_id ', i_id, ' begin caculate')
            for u_id, i_score in i_id_ratings:
                # 开始预测u_id给i_id打分
                
                # 临时存储u打分了的item与这个item的相似度
                # item -> sim
                sim_item_dict = {}
                
                # 存储u_id给sim_item打的分
                u_sim_score = {}

                # 预测打分 以baseline作下界 son和mother代表分子与分母
                predict_score = self.caculate_bxi(u_id, i_id)
                predict_score_son = 0
                predict_score_mother = 0
                # u_id 给其他的item打分过的
                for [sim_item, sim_item_rating] in self.train_user_data[u_id]:
                    flag = 1
                    if i_id in self.simmap and sim_item in self.simmap[i_id]:
                        sim_res = self.simmap[i_id][sim_item]
                        flag = 0
                    if sim_item in self.simmap and i_id in self.simmap[i_id]:
                        sim_res = self.simmap[sim_item][i_id]
                        flag = 0
                    if flag ==1 :
                        sim_res = self.caculate_person_sim(i_id, sim_item)
                    if sim_res!=0:
                        
                        sim_item_dict[sim_item] = sim_res
                        u_sim_score[sim_item] = sim_item_rating
                
                # 计算的相似度进行排序
                sim_item_dict = sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True)
                # 取最相似的50个
                count = 0
                for (most_sim, person_sim) in sim_item_dict:
                    predict_score_son += person_sim * (u_sim_score[most_sim] - self.caculate_bxi(u_id, most_sim))
                    predict_score_mother += person_sim
                    count+=1
                    if count == 50:
                        break
                predict_score += predict_score_son / predict_score_mother
                # 把预测的用于计算RMSE
                sum_RMSE += (i_score - predict_score)**2
            print('i_id ', i_id, ' caculate over')
            sum_n += len(i_id_ratings)
            print('begin store')
            self.save_params()
            print('store over')
            
            
        
        sum_RMSE = np.sqrt(sum_RMSE/sum_n)
        return sum_RMSE
    
    def save_params(self):
        #使用pickle
        CF_path="./CF/"
        save_pickle(self.simmap, CF_path+'simmap.pkl')
        
    def test_write(self):
        # test_data 中的全部都是真实u_id和i_id
        predict_score_list = defaultdict(list)
        for u_id, all_items in self.test_data.items():
            for i_id in all_items:
                # 获取u_id映射之后的id
                u_f_id = self.user_idx[u_id]
                if i_id in self.item_idx:
                    # 获取i_id映射之后的id
                    i_f_id = self.item_idx[i_id]
                    # CF方式预测
                    sim_item_dict = {}
                    u_sim_score = {}
                    # 预测打分 以baseline作下界 son和mother代表分子与分母
                    predict_score = self.caculate_bxi(u_f_id, i_f_id)
                    predict_score_son = 0
                    predict_score_mother = 0
                    # u_id 给其他的item打分过的
                    for [sim_item, sim_item_rating] in self.train_user_data[u_f_id]:
                        flag = 1
                        if i_f_id in self.simmap and sim_item in self.simmap[i_f_id]:
                            sim_res = self.simmap[i_f_id][sim_item]
                            flag = 0
                        if sim_item in self.simmap and i_f_id in self.simmap[i_f_id]:
                            sim_res = self.simmap[sim_item][i_f_id]
                            flag = 0
                        if flag == 1:
                            sim_res = self.caculate_person_sim(i_f_id, sim_item)
                        if sim_res!=0:
                            sim_item_dict[sim_item] = sim_res
                            u_sim_score[sim_item] = sim_item_rating
                
                    # 计算的相似度进行排序, 计算时只取最多30个
                    sim_item_dict = sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True)
                    count = 0
                    for (most_sim, person_sim) in sim_item_dict:
                        predict_score_son += person_sim * (u_sim_score[most_sim] - self.caculate_bxi(u_f_id, most_sim))
                        predict_score_mother += person_sim
                        count+=1
                        if count == 50:
                            break
                        
                    if predict_score_mother !=0 and predict_score_son != 0:
                        predict_score += predict_score_son / predict_score_mother
                    
                    if predict_score<0:
                        predict_score = 0
                    if predict_score>100:
                        predict_score = 100
                        
                    predict_score_list[u_id].append([i_id, predict_score])

                else:
                    predict_score = self.miu + self.bx[self.user_idx[u_id]]
                    
                    if predict_score<0:
                        predict_score = 0
                    if predict_score>100:
                        predict_score = 100
                    predict_score_list[u_id].append([i_id, predict_score])
        
        # 写入指定路径
        with open(test_predict_result_path, 'w') as f:
            for u_id, item_ratings in predict_score_list.items():
                num = len(item_ratings)
                f.write(str(u_id)+'|'+str(num)+'\n')
                for i_id, i_rating in item_ratings:
                    f.write(str(i_id)+' '+str(i_rating)+'\n')
                    
                    
if __name__ == '__main__':
    mycf = my_basicCF()
    # train数据集上计算RMSE
    RMSE = mycf.train()
    print('RMSE: ', RMSE)
    # 预测test数据集
    print('begin predict test_data')
    mycf.test_write()
    print('predict test_data over')
        
                    
        
                    








                    
    