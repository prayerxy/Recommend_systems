from collections import defaultdict
from utils import load_pickle, save_pickle, get_mean
import numpy as np
# 如果预测时item从来没有被打过分，舍弃person，就将其余弦相似度作为相似度

# user_item_data和item_user_data路径
user_item_pkl = './mypickle/user_item_matrix.pkl'
item_user_pkl = './mypickle/item_user_matrix.pkl'

# 存储bias路径
bx_path = './mypickle/default_bx.pkl'
bi_path = './mypickle/default_bi.pkl'

# test数据pickle路径
test_data_path = './mypickle/test_matrix.pkl'

# item_attrs路径
item_attrs_path = './mypickle/item_attrs.pkl'

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
        self.user_item_data = load_pickle(user_item_pkl)
        self.item_user_data = load_pickle(item_user_pkl)
        self.test_data = load_pickle(test_data_path)
        self.item_attrs = load_pickle(item_attrs_path)


    def caculate_bxi(self, user_x, item_i):
        return self.miu + self.bx[user_x] + self.bi[item_i]

    # 传入的是映射的i_id和sim_item
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
        for [user_i, user_i_rating] in self.item_user_data[i_id]:
            # 给i_id打过分的所有的
            sim_dis_left += (user_i_rating - avg_item)**2
            for [user_s, user_s_rating] in self.item_user_data[sim_item]:
                if user_i == user_s:
                    sim_son += (user_i_rating - avg_item)*(user_s_rating - avg_sim_item)
                # 给sim_item打过分的所有的
                sim_dis_right += (user_s_rating - avg_sim_item)**2
                        
        if sim_dis_left!=0 and sim_dis_right!=0:
            sim_res = sim_son/((sim_dis_left * sim_dis_right)**(1/2))
        # 否则sim_res仍然为0
        return sim_res
        
        
    
    def train(self):
        # 采取分块计算策略
        sum_RMSE = 0
        # 统计打分个数
        sum_n = 0
        for u_id, i_id_ratings in self.user_item_data.items():
            for i_id, i_score in i_id_ratings:
                # 开始预测u_id给i_id打分
                # 临时存储u打分了的item与这个item的相似度
                # item -> sim
                sim_item_dict = {}

                # 预测打分 以baseline作下界 son和mother代表分子与分母
                predict_score = self.caculate_bxi(u_id, i_id)
                predict_score_son = 0
                predict_score_mother = 0
                # u_id 给其他的item打分过的
                for [sim_item, sim_item_rating] in self.user_item_data[u_id]:
                    sim_res = self.caculate_person_sim(i_id, sim_item)
                    if sim_res!=0:
                        sim_item_dict[sim_item] = sim_res
                
                # 计算的相似度进行排序, 计算时只取最多30个
                sim_item_dict = sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True)
                count = 0
                for most_sim, person_sim in sim_item_dict.items():
                    if count>30:
                        break
                    predict_score_son += person_sim * (self.item_user_data[most_sim][u_id] - self.caculate_bxi(u_id, most_sim))
                    predict_score_mother += person_sim

                predict_score += predict_score_son / predict_score_mother
                # 把预测的用于计算RMSE
                sum_RMSE += (i_score - predict_score)**2
            sum_n += len(i_id_ratings)
        
        sum_RMSE = np.sqrt(sum_RMSE/sum_n)
        return sum_RMSE
    
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
                    # 预测打分 以baseline作下界 son和mother代表分子与分母
                    predict_score = self.caculate_bxi(u_f_id, i_f_id)
                    predict_score_son = 0
                    predict_score_mother = 0
                    # u_id 给其他的item打分过的
                    for [sim_item, sim_item_rating] in self.user_item_data[u_f_id]:
                        sim_res = self.caculate_person_sim(i_f_id, sim_item)
                        if sim_res!=0:
                            sim_item_dict[sim_item] = sim_res
                
                    # 计算的相似度进行排序, 计算时只取最多30个
                    sim_item_dict = sorted(sim_item_dict.items(), key=lambda x: x[1], reverse=True)
                    count = 0
                    for most_sim, person_sim in sim_item_dict.items():
                        if count>30:
                            break
                        predict_score_son += person_sim * (self.item_user_data[most_sim][u_f_id] - self.caculate_bxi(u_f_id, most_sim))
                        predict_score_mother += person_sim
                    if predict_score_mother !=0 and predict_score_son != 0:
                        predict_score += predict_score_son / predict_score_mother
                    predict_score *= 10
                    if predict_score<0:
                        predict_score = 0
                    if predict_score>100:
                        predict_score = 100
                        
                    predict_score_list[u_id].append([i_id, predict_score])

                else:
                    predict_score = self.miu + self.bx[self.user_idx[u_id]]
                    # 看能否通过attribute计算余弦相似度
                    for [sim_item, sim_item_rating] in self.user_item_data[u_f_id]:
                        
                        
                    predict_score*=10
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
        
                    
        
                    








                    
    