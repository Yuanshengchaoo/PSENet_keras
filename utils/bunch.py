"""Lightweight Bunch compatibility helper allowing attribute access for dict keys."""

# 简单的 bunch 兼容实现，支持属性访问：obj.key 等价于 obj["key"]
class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 关键：dict 的 __dict__ 指向自己，这样就可以用点号访问
        self.__dict__ = self

    @classmethod
    def from_dict(cls, d):
        """可选：模仿原库的 fromDict 接口（如果项目有用到）"""
        return cls(d)
