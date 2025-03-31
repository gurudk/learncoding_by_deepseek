from gensim import corpora, models

# 示例文档
documents = [["machine", "learning", "data", "algorithm"],
             ["biology", "gene", "dna", "cell"],
             ["machine", "dna", "data", "cell"]]

# 创建词典和文档-词矩阵
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# 训练LDA模型
lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# 输出主题
print(lda.print_topics())