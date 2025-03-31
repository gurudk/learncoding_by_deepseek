
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

# --------------------
# 1. 准备示例数据
# --------------------
# 每个句子由单词列表组成，每个单词对应一个标签（BIO标注格式）
# 标签含义：B-PER（人名开头），I-PER（人名中间），B-LOC（地名开头），O（非实体）

sentences = [
    {
        "words": ["John", "Smith", "works", "in", "New", "York", "at", "Google"],
        "labels": ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "O", "B-ORG"]
    },
    {
        "words": ["Apple", "is", "based", "in", "Cupertino", ",", "California"],
        "labels": ["B-ORG", "O", "O", "O", "B-LOC", "O", "B-LOC"]
    }
]


# --------------------
# 2. 特征提取函数
# --------------------
def word2features(sent, i):
    word = sent[i]
    features = {
        "word": word,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],  # 后3个字符（捕捉常见后缀）
        "word[:3]": word[:3],  # 前3个字符（捕捉常见前缀）
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }

    # 添加上下文特征（前一个词和后一个词）
    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            "prev_word": prev_word,
            "prev_word.lower()": prev_word.lower(),
        })
    else:
        features["BOS"] = True  # 句子开头

    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            "next_word": next_word,
            "next_word.lower()": next_word.lower(),
        })
    else:
        features["EOS"] = True  # 句子结尾

    return features


# 将整个句子转换为特征序列
def prepare_features(data):
    X, y = [], []
    for sentence in data:
        words = sentence["words"]
        labels = sentence["labels"]
        X_sent = [word2features(words, i) for i in range(len(words))]
        X.append(X_sent)
        y.append(labels)
    return X, y


X, y = prepare_features(sentences)

# --------------------
# 3. 训练CRF模型
# --------------------
# 划分训练集和测试集（此处示例数据简单，实际需更多数据）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化CRF模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',  # 优化算法
    c1=0.1,  # L1正则化系数
    c2=0.1,  # L2正则化系数
    max_iterations=100,
    all_possible_transitions=True  # 允许所有可能的标签转移
)

# 训练模型
crf.fit(X_train, y_train)

# --------------------
# 4. 评估与预测
# --------------------
# 预测测试集
y_pred = crf.predict(X_test)

# 输出分类报告
labels = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "O"]
print(metrics.flat_classification_report(y_test, y_pred, labels=labels))

# 预测新句子
new_sentence = ["Tim", "Cook", "visits", "Paris"]
X_new = [word2features(new_sentence, i) for i in range(len(new_sentence))]
pred_labels = crf.predict_single(X_new)

print("\n预测结果：")
for word, label in zip(new_sentence, pred_labels):
    print(f"{word} -> {label}")