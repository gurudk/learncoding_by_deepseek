from pyswip import Prolog

# 初始化Prolog引擎
prolog = Prolog()

# 加载Prolog文件或直接声明规则
prolog.assertz("father(michael, john)")
prolog.assertz("father(michael, mary)")
prolog.assertz("father(john, ann)")

# 执行查询
print(list(prolog.query("father(michael, Y)")))
# 输出: [{'X': 'john'}, {'X': 'mary'}]

# 可以加载外部.pro文件
# prolog.consult("path_to_your_prolog_file.pl")