from collections import defaultdict
import numpy as np
from utils import load_pickle, get_mean

# bx路径 and bi路径
bx_path = './mypickle/default_bx.pkl'
bi_path = './mypickle/default_bi.pkl'

# test数据pickle路径
test_data_path = './mypickle/test_matrix.pkl'
# 映射路径
users_idx = './mypickle/users.pkl'
items_idx = './mypickle/items.pkl'

# user_item路径
user_item_pkl = './mypickle/user_item_matrix.pkl'
# test预测结果写入路径
test_predict_result_path = './Result/result_baseline.txt'

# 全局评分均值
miu = get_mean()



# 定义baseline算法类
class BL:
    def __init__(self):
        self.bx = load_pickle(bx_path)  # 用户偏置
        self.bi = load_pickle(bi_path)  # 物品偏置
        self.user_idx = load_pickle(users_idx)
        self.item_idx = load_pickle(items_idx)
        self.user_item_data = load_pickle(user_item_pkl)
        self.test_data = load_pickle(test_data_path)

    def predict(self, u_id, i_id):
        # u_id和i_id全部都是映射的
        predict_score = miu + self.bx[u_id] + self.bi[i_id]
        return predict_score
    
    def caculate_RMSE(self):
        # 计算分子差的平方
        sum_loss = 0.0
        # 分母计数
        n = 0
        for u_id, i_id_ratings in self.user_item_data.items():
            for i_id, i_score in i_id_ratings:
                sum_loss += (i_score - self.predict(u_id, i_id))**2
                n+=1
        
        RMSE = np.sqrt(sum_loss/n)
        return RMSE
    
    def test_write(self):
        # 存储结果使用字典列表
        predict_score = defaultdict(list)
        for u_id, all_items in self.test_data.items():
            for i_id in all_items:
                # 判断i_id对应的物品是否被打过分
                if i_id not in self.item_idx:
                    # 没有被打过分
                    tmp_score = miu + self.bx[self.user_idx[u_id]]
                    if tmp_score > 100:
                        tmp_score = 100
                    if tmp_score < 0:
                        tmp_score = 0
                else:
                    tmp_score = self.predict(self.user_idx[u_id], self.item_idx[i_id])
                    if tmp_score > 100:
                        tmp_score = 100
                    if tmp_score < 0:
                        tmp_score = 0
                
                predict_score[u_id].append([i_id, tmp_score])
        
        # 开始写入到指定路径
        with open(test_predict_result_path, 'w') as f:
            for u_id, item_ratings in predict_score.items():
                num = len(item_ratings)
                f.write(str(u_id)+'|'+str(num)+'\n')
                for i_id, i_rating in item_ratings:
                    f.write(str(i_id)+' '+str(i_rating)+'\n')
        


if __name__ == '__main__':
    baseline = BL()
    # 计算RMSE
    RMSE = baseline.caculate_RMSE()
    print('RMSE: ', RMSE)

    # 预测test数据集
    print('begin predict test_data')
    baseline.test_write()
    print('predict test_data over')



    

    
