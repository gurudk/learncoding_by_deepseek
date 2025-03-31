from durable.lang import *

# 定义规则集
with ruleset('zoo'):
    # 当添加新动物时触发
    @when_all(+m.entity == 'animal', +m.name, +m.species, +m.age)
    def add_animal(c):
        print(f"动物 {c.m.name} ({c.m.species}) 已加入，年龄为 {c.m.age}。")

        # 检查是否需要特别关注（例如，年老或濒危物种）
        if c.m.age > 15 or c.m.species in ['华南虎', '大熊猫']:
            print(f"{c.m.name} 需要特别关照！")


    # 当更新动物信息时触发
    @when_all(m.entity == 'animal', m.update == True)
    def update_animal(c):
        print(f"动物 {c.m.name} 的信息已更新。新的年龄为 {c.m.age}。")


    # 当移除动物时触发
    @when_all(m.entity == 'animal', m.remove == True)
    def remove_animal(c):
        print(f"动物 {c.m.name} 已从动物园中移除。")

# 添加动物到动物园
assert_fact('zoo', {'entity': 'animal', 'name': '大黄', 'species': '狮子', 'age': 8})
assert_fact('zoo', {'entity': 'animal', 'name': '小花', 'species': '华南虎', 'age': 16})

# 更新动物信息
update_state('zoo', {'entity': 'animal', 'name': '大黄', 'update': True, 'age': 9})

# # 移除动物
# retract_fact('zoo', {'entity': 'animal', 'name': '小花', 'remove': True})