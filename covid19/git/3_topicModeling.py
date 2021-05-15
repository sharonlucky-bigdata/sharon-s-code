# C. Lyndon Luo, 2021-1-25
# 深圳专属语料的主题模型
import warnings
warnings.filterwarnings("ignore")

import re
import tqdm
import pyLDAvis
import pyLDAvis.gensim
import pandas as pd
from collections import Counter, defaultdict
from gensim import corpora, models

# 用来存储语料的列表（原始语料）
texts = list()
# 已经有处理好的分词结果
with open("/Users/lishuang/covid19/corpus_token.txt", "r") as f1:
    for line in f1.readlines():
        texts.append(line.split("\t")[1].strip().split())
# print(texts)

# 第一步：建立词典
dictionary = corpora.Dictionary(texts)
# # 文章数目
print(dictionary.num_docs)
# # 所有词语的数量
print(dictionary.num_pos)
print(len(dictionary))
# # 必要步骤：对字典进行过滤（词语数量 < 3, 词频 > 0.85）
dictionary.filter_extremes(no_below = 3, no_above = 0.85)
# # 再次打印文章数目、所有词语的数量
print(dictionary.num_docs)
print(dictionary.num_pos)
print(len(dictionary))
# # 存储词典
dictionary.save("./dictionary.dict")
# print(dictionary)
# # 查看具体的词语 - ID映射关系
# print(dictionary.token2id)
#
# # 第二步：将文档转换为向量
corpus = [dictionary.doc2bow(text) for text in texts]
# # 存储向量
corpora.MmCorpus.serialize("corpus.mm", corpus)

# 第三步：TF-IDF转换（之前的两步操作可以注释掉，因为已经把词典和语料序列化了）
# dictionary = corpora.Dictionary.load("./dictionary.dict")
# corpus = corpora.MmCorpus("./corpus.mm")
# print(len(corpus))
# tfidf = models.TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]
# # 观察转换前后的对比
# for i in corpus[:10]:
#     print(i)
# for i in corpus_tfidf[:10]:
#     print(i)

# # # 第四步：使用HDP，基于非参贝叶斯，不需要事先确定主题数量，可以作为初步探索（结果：给出150个主题）
# model = models.HdpModel(corpus_tfidf, id2word=dictionary)
# print(model.get_topics().shape)
# print(model.show_topics(num_topics=30))

# 第五步：使用 C_V coherence 确定主题数（第四步可以注释掉，一轮训练时间约为 2.5h）
def cv_score(corpus, dict_, k, alpha, eta):
    lda_model = models.LdaMulticore(corpus = corpus,
                                    id2word = dict_,
                                    num_topics = k,
                                    alpha = alpha,
                                    eta = eta,
                                    random_state = 100,
                                    chunksize = 100,
                                    passes = 10,
                                    per_word_topics = True)
    coherence = models.CoherenceModel(model = lda_model,
                                      texts = texts,
                                      corpus = corpus,
                                      dictionary = dict_,
                                      coherence = "c_v") # u_mass, c_v, c_uci, c_npmi
    return coherence.get_coherence()
#
# # # 主体数量 k 的范围，共 19 个取值
# topics_range = list(range(2, 21, 1))
# # # alpha (topics' probability) 的范围，共 2 个取值
# alpha = ["symmetric", "asymmetric"]
# # # eta (word probability) 的范围，共 2 个取值
# eta = ["symmetric", "auto"]
#
# print("进度：")
# model_results = {"topics": list(), "alpha": list(), "eta": list(), "cv_score": list()}
# if 1 == 1:
#     bar = tqdm.tqdm(total = 19 * 2 * 2)
#     for k in topics_range:
#         for a in alpha:
#             for e in eta:
#                 cv = cv_score(corpus = corpus, dict_ = dictionary, k = k, alpha = a, eta = e)
#                 model_results["topics"].append(k)
#                 model_results["alpha"].append(a)
#                 model_results["eta"].append(e)
#                 model_results["cv_score"].append(cv)
#                 bar.update(1)
#     pd.DataFrame(model_results).to_csv("./tm_results.csv", index = False)
#     bar.close()

# 第六步：选择参数组合，获得主题对应的词语，并进行可视化
# pattern_word = re.compile("\"(.*?)\"") #正则提取主题，有负值不做正则
k = 30
alpha = "symmetric"#可选
eta = "auto"
print(cv_score(corpus=corpus, dict_ = dictionary, k=30, alpha="symmetric", eta="auto"))
final_model = models.LdaMulticore(corpus = corpus,
                                  id2word = dictionary,
                                  num_topics = k,
                                  alpha = alpha,
                                  eta = eta,
                                  random_state = 100,
                                  chunksize = 100,
                                  passes = 10,
                                  per_word_topics = True)
for idx, item in enumerate(final_model.print_topics(num_topics = -1, num_words = 30)):
    print("Topic %s has following keywords: "%(idx))
    patterns=re.findall("\"(.*?)\"",str(item),re.S)
    print(patterns)
    # print(cv_score(corpus=corpus, dict_ = dictionary, k=30, alpha="symmetric", eta="auto"))
# 使用 pyLDAvis 进行可视化
viz = pyLDAvis.gensim.prepare(final_model, corpus, dictionary)
pyLDAvis.save_html(viz, "./tm_viz.html")
#命名生成的html
# 修改网页中的 3 处调用
with open("./tm_viz_new.html", "w") as t1:
    with open("./tm_viz.html", "r") as f2:
        webpage = f2.read()
        # 将 css js 文件存放在本地，避免加载堵塞
        webpage = webpage.replace("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css",
                                  "ldavis.v1.0.0.css")
        webpage = webpage.replace("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js",
                                  "ldavis.v1.0.0.js")
        # 需要修正 d3.js 调用，回归老版本
        webpage = webpage.replace("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js",
                                  "https://d3js.org/d3.v3.js")
    t1.write(webpage)


#### 加入于2021年2月21日，文档对应的主题
# 需要传入的参数：词袋模型，也就是之前定义的corpus；还可以指定minimum_probability，先设置为主题数量的倒数，也即1/4=0.25
# 包装输出的结果到一个列表中
doc_topics = list()
for i in final_model.get_document_topics(corpus, minimum_probability=float(1/k)):
    # i包含[(主题编号，对应概率)]
    # print([j[0] for j in i])
    doc_topics.append([j[0] for j in i])
#
# # 将机构的名称存入另一个列表中
# # 结果：文章编号：机构名称
# temp1 = dict()
# with open("/Volumes/Lyndon/yanlong/文件新旧名称映射_深圳专属（旧）/mapping_with_date_1229.txt", "r") as f3:
#     for line in f3.readlines():
#         temp1[line.strip().split("\t")[0]] = line.strip().split("\t")[-1].split("/")[-2]
# # print(temp1)
#
# # 机构名称
# temp2 = list()
# with open("/Volumes/Lyndon/yanlong/深圳重命名（复制2020-12-29）分析结果/corpus_token.txt", "r") as f4:
#     for line in f4.readlines():
#         if line.strip().split("\t")[0] in temp1.keys():
#             temp2.append(temp1[line.strip().split("\t")[0]])
# # print(temp2)
#
# # 计算机构发布的文章数量
# org_doc_num = dict(Counter(temp2))
#
# # 构建网络
# with open("./org_topic_net.csv", "w") as t2:
#     t2.write("source,target\n")
#     for item in zip(doc_topics, temp2):
#         # print(item)
#         if len(item[0]) > 1:  # item[0]大于1表示有多个主题
#             for id_ in item[0]:
#                 # print(id_, item[1])
#                 t2.write("Topic" + str(id_) + "," + item[1])
#                 t2.write("\n")
#         else: # 表示仅有一个主题
#             # print(item[0][0], item[1])
#             t2.write("Topic" + str(item[0][0]) + "," + item[1])
#             t2.write("\n")
#
# # 计算机构对应的主题概率
# org_topic_net = pd.read_csv("./org_topic_net.csv", header=0, index_col=None)
# observed = pd.crosstab(org_topic_net.target, org_topic_net.source)
# # print(observed)
# with open("./org_topic_net_perc.csv", "w") as t3:
#     t3.write("source,target,weight\n")
#     for row in observed.index:
#         for col in observed.columns:
#             if observed.loc[row, col] != 0:
#                 t3.write(row + "," + col + "," + str((observed.loc[row, col] / org_doc_num[row]) * 100))
#                 t3.write("\n")


# #### 加入于2021年3月11日，只寻找文档对应的最主要主题
# doc_topics = list()
# for i in final_model.get_document_topics(corpus):
#     i_ = sorted(i, key=lambda x: x[1], reverse=True)
#     doc_topics.append(i_[0][0])
#
# # 将机构的名称存入另一个列表中
# # 结果：文章编号：机构名称
# temp1 = dict()
# with open("/Volumes/Lyndon/yanlong/文件新旧名称映射_深圳专属（旧）/mapping_with_date_1229.txt", "r") as f3:
#     for line in f3.readlines():
#         temp1[line.strip().split("\t")[0]] = line.strip().split("\t")[-1].split("/")[-2]
# # print(temp1)
#
# # 机构名称
# temp2 = list()
# with open("/Volumes/Lyndon/yanlong/深圳重命名（复制2020-12-29）分析结果/corpus_token.txt", "r") as f4:
#     for line in f4.readlines():
#         if line.strip().split("\t")[0] in temp1.keys():
#             temp2.append(temp1[line.strip().split("\t")[0]])
# # print(temp2)
#
# # 计算机构发布的文章数量
# org_doc_num = dict(Counter(temp2))
#
# # 构建网络
# with open("./org_topic_net_1.csv", "w") as t2:
#     t2.write("source,target\n")
#     for item in zip(doc_topics, temp2):
#         t2.write("Topic" + str(item[0]) + "," + item[1])
#         t2.write("\n")
#
# # 计算机构对应的主题概率
# org_topic_net = pd.read_csv("./org_topic_net_1.csv", header=0, index_col=None)
# observed = pd.crosstab(org_topic_net.target, org_topic_net.source)
# # print(observed)
# with open("./org_topic_net_perc_1.csv", "w") as t3:
#     t3.write("source,target,weight\n")
#     for row in observed.index:
#         for col in observed.columns:
#             if observed.loc[row, col] != 0:
#                 t3.write(row + "," + col + "," + str((observed.loc[row, col] / org_doc_num[row]) * 100))
#                 t3.write("\n")

# #### 加入于2021年3月18日，只寻找机构对应的最主要主题
# # 只需要使用`org_topic_net_perc_one2one.csv`一个文件即可
# df = pd.read_csv("./org_topic_net_perc_one2one.csv", header=0, index_col=None)
# pool = defaultdict(list)
#
# for idx in range(df.shape[0]):
#     pool[df.loc[idx, "source"]].append((df.loc[idx, "target"], df.loc[idx, "weight"]))
#
# with open("./org_topic_net_prominent.csv", "w", encoding="utf-8") as t4:
#     t4.write("source,target,weight\n")
#     for k, v in dict(pool).items():
#         print(k, sorted(v, key=lambda x: x[1], reverse=True)[0][0])
#         t4.write(k + "," + sorted(v, key=lambda x: x[1], reverse=True)[0][0] + ",1" + "\n")

