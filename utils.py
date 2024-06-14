from collections import defaultdict
import pickle as pkl

import numpy as np

train_path='./data/train.txt'
test_path='./data/test.txt'
iterm_attr_path='./data/itemAttribute.txt'
pickle_path='./mypickle/'

'''
数量统计
训练集:
items:455691
users:19835
ratings: 5001507

测试集:
items: 28242
users: 19835
ratings: 119010

item_attribute: 624961
'''

#将user和item映射到自然数0开始
def load_index():
    #首先把items以及users映射到自然数
    with open(train_path,'r') as f:
        #一行行读入
        items=set()
        users=set()
        for line in f:
            user,num=line.split('|')
            user=int(user)
            num=int(num)
            users.add(user)
            for _ in range(num):
                item,_=map(int,f.readline().split())
                items.add(item)
    items=sorted(list(items))
    users=sorted(list(users))
    print('items:',len(items))
    print('users:',len(users))
    items={i:idx for idx,i in enumerate(items)}
    users={u:idx for idx,u in enumerate(users)}
    #保存映射关系
    with open(pickle_path+'items.pkl','wb') as f:
        pkl.dump(items,f)
    with open(pickle_path+'users.pkl','wb') as f:
        pkl.dump(users,f)

def load_true_item():
    #items反着映射回去
    with open(pickle_path+'items.pkl','rb') as f:
        items=pkl.load(f)
    items={idx:i for i,idx in items.items()}
    return items

def create_matrix():
    #创建user-item和item-user的矩阵
    #采取defaultdict(list)的形式  关键字映射到list
    with open(pickle_path+'items.pkl','rb') as f:
        items=pkl.load(f)
    with open(pickle_path+'users.pkl','rb') as f:
        users=pkl.load(f)
    #创建user-item矩阵
    user_item_matrix=defaultdict(list)
    with open(train_path,'r') as f:
        for line in f:
            user,num=line.split('|')
            user=int(user)
            num=int(num)
            for _ in range(num):
                item,rate=f.readline().split()
                item=int(item)
                rate=float(rate)
                user_item_matrix[users[user]].append([items[item],rate])
    #保存user-item矩阵
    with open(pickle_path+'user_item_matrix.pkl','wb') as f:
        pkl.dump(user_item_matrix,f)
    #创建item-user矩阵
    item_user_matrix=defaultdict(list)
    with open(train_path,'r') as f:
        for line in f:
            user,num=line.split('|')
            user=int(user)
            num=int(num)
            for _ in range(num):
                item,rate=f.readline().split()
                item=int(item)
                rate=float(rate)
                item_user_matrix[items[item]].append([users[user],rate])
    #保存item-user矩阵
    with open(pickle_path+'item_user_matrix.pkl','wb') as f:
        pkl.dump(item_user_matrix,f)

def create_test_matrix():
    test_matrix=defaultdict(list)
    with open(test_path,'r') as f:
        for line in f:
            user,num=line.split('|')
            user=int(user)
            num=int(num)
            for _ in range(num):
                item=f.readline().split()[0]
                item=int(item)
                #test无rate
                test_matrix[user].append(item)
    with open(pickle_path+'test_matrix.pkl','wb') as f:
        pkl.dump(test_matrix,f)


def get_mean():
    with open(pickle_path+'user_item_matrix.pkl','rb') as f:
        user_item_matrix=pkl.load(f)
    sum=0.0
    count=0
    for _,items in user_item_matrix.items():
        for _,rate in items:
            sum+=rate
            count+=1
    return float(sum/count)

def create_bx():
    with open(pickle_path+'user_item_matrix.pkl','rb') as f:
        user_item_matrix=pkl.load(f)
    mean=get_mean()
    # print(len(user_item_matrix))
    bx=np.zeros(len(user_item_matrix),dtype=np.float64)
    for user,items in user_item_matrix.items():
        sum=0.0
        count=0
        for _,rate in items:
            sum+=rate
            count+=1
        bx[user]=float(sum/count)-mean
    #储存至mypickle
    with open(pickle_path+'default_bx.pkl','wb') as f:
        pkl.dump(bx,f)

def create_bi():
    with open(pickle_path+'item_user_matrix.pkl','rb') as f:
        item_user_matrix=pkl.load(f)
    mean=get_mean()
    # print(len(item_user_matrix))
    bi=np.zeros(len(item_user_matrix),dtype=np.float64)
    for item,users in item_user_matrix.items():
        sum=0.0
        count=0
        for _,rate in users:
            sum+=rate
            count+=1
        bi[item]=float(sum/count)-mean
    #储存至mypickle
    with open(pickle_path+'default_bi.pkl','wb') as f:
        pkl.dump(bi,f)

def get_bi():
    with open(pickle_path+'default_bi.pkl','rb') as f:
        return pkl.load(f)

def get_bx():
    with open(pickle_path+'default_bx.pkl','rb') as f:
        return pkl.load(f)
    

def get_item_attrs():
    # 存储 item -> [attr1, attr2]
    item_attrs = defaultdict(list)
    # 如果物品被打过分，存储映射的i_id，否则没打过分的存真实的i_id
    max_item_id = 0
    with open(iterm_attr_path, 'r') as f:
        for line in f:
            item, attr1, attr2 = line.strip().split('|')
            item = int(item)
            if attr1 == 'None':
                attr1 = 0
            else :
                attr1 = float(attr1)
            if attr2 == 'None':
                attr2 = 0
            else :
                attr2 = float(attr2)
            if item>max_item_id:
                # 补全中间的
                for i in range(max_item_id+1,item):
                    item_attrs[i] = [0,0,0]
                max_item_id = item
            item_attrs[item].append(attr1)
            item_attrs[item].append(attr2)
            norm = np.sqrt(attr1*attr1+attr2*attr2)
            item_attrs[item].append(norm)
            
    print('max_item_id: ',max_item_id)
    # 保存
    with open(pickle_path+'item_attrs.pkl','wb') as f:
        pkl.dump(item_attrs,f)

def load_pickle(path):
    with open(path,'rb') as f:
        return pkl.load(f)

def save_pickle(obj,path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)


if __name__ == '__main__':
    print("start...")
    load_index()
    create_matrix()
    create_bi()
    create_bx()
    create_test_matrix()
    get_item_attrs()
    print("end...")


        

