import pandas as pd
import numpy as np
import random as rd
import jieba
import wordcloud
from scipy.spatial import distance
from tqdm import tqdm

MAXDIS = 1e8

def CaculateWordsFrequency(file_name, stopfile_name):#计算词频
    df = pd.read_csv(file_name, encoding='utf-8', usecols=[0])#读取文件
    comments = list(df['content'])
    stop_f = open(stopfile_name, 'r', encoding='utf-8')#导入分词表
    jieba.load_userdict(stopfile_name)
    stop_words = set()
    for line in stop_f:
        line = line.rstrip('\n')
        if len(line):
            stop_words.add(line)
    stop_f.close()
    counts = {}#分词+记录频率
    seg_list = []
    documents = []
    for content in comments:
        cur_seg_list = jieba.lcut(content)
        seg_list.append(cur_seg_list)
        documents.append(' '.join(cur_seg_list))
        for seg in cur_seg_list:
            if seg not in stop_words:
                counts[seg] = counts.get(seg, 0) + 1
    counts = dict(sorted(counts.items(), key=lambda dc: dc[1], reverse=True))
    print('TOP10: {}\n'.format(list(counts.keys())[:10]))
    dic_word = {'word':counts.keys(), 'frequcency':counts.values()}
    pd.DataFrame(dic_word).to_csv('word_counts.csv')
    stop_f.close()

    return dic_word, counts, documents, seg_list, stop_words, comments


def OutputWordCloud(counts, n):#绘制词云图
    wordpng = wordcloud.WordCloud(background_color='white', height=700, width=1000, font_path='C:\Windows\Fonts\simHei.ttf')#绘制词云
    wordpng.generate(" ".join(list(counts.keys())[:n]))
    wordpng.to_file("result.png")

def OutputMainWord(counts):#打印关键词
    word_list = [key for key, value in counts.items() if value > 3]
    word_list = list(set(word_list))
    word_id = {}
    for i in range(len(word_list)):
        word_id[word_list[i]] = i
    return set(word_list), word_id

def CaculateEuclid_distance(x1, x2):#计算欧氏距离
    dis=distance.euclidean(np.array(x1), np.array(x2))
    return dis

def CaculateCosine_distance(x1, x2):#计算余弦相似度
    result = distance.cosine(np.array(x1), np.array(x2))
    return result

def RandomComment(seg_list):#随机选取弹幕
    id = rd.randint(0, len(seg_list) - 1)
    random_vector = [0] * (len(main_word) + 5)
    for word in seg_list[id]:
        if word in main_word:
            random_vector[word_id[word]] = 1
    return comments[id], random_vector

dic_words, counts, documents, seg_list, stop_words, comments = CaculateWordsFrequency('D:\Project\Python\week2\danmuku.csv', 'D:\Project\Python\week2\stopwords_list.txt')
OutputWordCloud(counts, 50)
main_word, word_id = OutputMainWord(counts)
print('main word:', len(main_word), end='')
for i in range(50):
    print(rd.choice(list(main_word)), end=',')
print()
random_list = []
for i in range(5):
    content, vector = RandomComment(seg_list)
    random_list.append(vector)
    print('random comment: ', i+1, ': ', content, sep='')
euclid_dis = [[CaculateEuclid_distance(c1, c2) for c2 in random_list] for c1 in random_list]
cos_dis = [[CaculateCosine_distance(c1, c2) for c2 in random_list] for c1 in random_list]
print('Euclid_distance:\n', euclid_dis)
print('Cosine_distance:\n', cos_dis)
