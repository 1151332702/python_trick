# -*- coding: utf-8 -*-
# @Time    : 2019/3/12 9:01
# @Author  : lilong
# @File    : singleton.py
# @Description:

# new 方法实现单例
# 因为重写__new__方法，所以继承至Singleton的类，在不重写__new__的情况下都将是单例模式
class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

a1 = Singleton()
a2 = Singleton()
print(a1 == a2)

# 元类实现单例
class Singleton(type):
    def __init__(self, *args, **kwargs):
        print('__init__')
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        print("__call__")
        if self.__instance is None:
            self.__instance = super(Singleton, self).__call__(*args, **kwargs)
        return self.__instance

class Foo(object, metaclass=Singleton):
    __metaclass__ = Singleton

foo1 = Foo()
foo2 = Foo()
print(Foo.__dict__)