# -*- coding: utf-8 -*-
# @Time    : 2019/3/11 19:13
# @Author  : lilong
# @File    : meta_class.py
# @Description:

class ObjectCreator(object):
    temp = 100
    pass

my_obj = ObjectCreator()
print(my_obj.__class__.temp)

# 类其实也是对象
# 可以将它赋值给一个变量，可以拷贝，可以为它增加属性，可以将它作为函数参数进行传递。
print(ObjectCreator)
ObjectCreator.new_attr = 'foo'
print(hasattr(ObjectCreator, 'new_attr'))
print(ObjectCreator.new_attr)

# 因为类也是对象，你可以在运行时动态的创建它们，就像其他任何对象一样。
# 首先，你可以在函数中创建类，使用class关键字即可。
def choose_class(name):
    if name == 'foo':
        class Foo(object):
            pass
        return Foo     # 返回的是类，不是类的实例
    else:
        class Bar(object):
            pass
        return Bar
my_class = choose_class('foo')
print(my_class)
print(my_class())

# type(类名, 父类的元组（针对继承的情况，可以为空），包含属性的字典（名称和值）)
class MyShinyClass(object):
    pass

my_class = type('MyShinyClass', (), {})
print(my_class)
print(my_class())

# 过一个具体的例子看看type是如何创建类
#构建目标代码
class Foo(object):
    bar = True
Foo = type('Foo', (), {'bar': True})
print(Foo())

#构建目标代码 继承父类：
class FooChild(Foo):
    pass
#使用type构建
FooChild = type('FooChild', (Foo,),{})
print(FooChild)

# 为Foochild类增加方法
def echo_bar(self):
    print(self.bar)
FooChild = type('FooChild', (Foo,), {'echo_bar': echo_bar})
print(hasattr(Foo, 'echo_bar'))
print(hasattr(FooChild, 'echo_bar'))
FooChild().echo_bar()

# 元类
# 元类就是用来创建这些类（对象）的，元类就是类的类
# str是用来创建字符串对象的类，而int是用来创建整数对象的类。type就是创建类对象的类
# 函数type实际上是一个元类
age = 10
print(age.__class__)
name = 's'
print(name.__class__)
def f():
    pass
print(f.__class__)
class Bar(object):
    pass
b = Bar()
print(b.__class__)
print(Bar.__class__)
# 对于任何一个__class__的__class__属性又是什么呢？ 是type
print(age.__class__.__class__)
print(b.__class__.__class__)

# 在写一个类的时候为其添加__metaclass__属性,定义了__metaclass__就定义了这个类的元类。
# 可以在__metaclass__中放置可以创建一个类的东西:
# type，或者任何使用到type或者子类化type的东西都可以
# 元类的主要目的就是为了当创建类时能够自动地改变类, 通常会为api做这些事情，希望可以创建符合当前上下文的类
# 例子： 模块里所有类的属性都是大写形式
# 元类会自动将你通常传给‘type’的参数作为自己的参数传入
# 使用函数当做元类
def upper_attr(future_class_name, future_class_parents, future_class_attr):
    attrs = ((name, value) for name, value in future_class_attr.items())
    uppercase_attr = {name.upper(): value for name, value in attrs}
    return type(future_class_name, future_class_parents, uppercase_attr)

class FooChild(metaclass=upper_attr):
    __metaclass__ = upper_attr
    bar = 'bilibili'
# 该类已经不含有bar属性  变成了BAR
print(hasattr(FooChild, 'bar'))
print(hasattr(FooChild, 'BAR'))
print(FooChild().BAR)

# 使用class来当做元类
class UpperAttrMetaClass(type):
    def __new__(upperattr_metaclass, future_class_name, future_class_parents, future_class_attr):
        attrs = ((name, value) for name, value in future_class_attr.items() if not name.startswith('__'))
        uppercase_attr = dict((name.upper(), value) for name, value in attrs)
        return type(future_class_name, future_class_parents, uppercase_attr)
class FooChild(metaclass=UpperAttrMetaClass):
    __metaclass__ = UpperAttrMetaClass
    bar = 'bilibili'
print('=======')
print(hasattr(FooChild, 'BAR'))

# OOP   这个upperattr_metaclass 其实一般写作cls   类方法的第一个参数都是该类的实例
class UpperAttrMetaclass(type):
    def __new__(upperattr_metaclass, future_class_name, future_class_parents, future_class_attr):
        attrs = ((name, value) for name, value in future_class_attr.items() if not name.startswith('__'))
        uppercase_attr = dict((name.upper(), value) for name, value in attrs)
        # 复用type.__new__方法
        # 这就是基本的OOP编程，没什么魔法。由于type是元类也就是类，因此它本身也是通过__new__方法生成其实例，只不过这个实例是一个类.
        return type.__new__(upperattr_metaclass, future_class_name, future_class_parents, uppercase_attr)

class FooChild(metaclass=UpperAttrMetaclass):
    __metaclass__ = UpperAttrMetaclass
    bar = 'bilibili'
print('=======')
print(hasattr(FooChild, 'BAR'))

# 调用父类的方法
class UpperAttrMetaclass(type):
    def __new__(cls, name, bases, dct):
        attrs = ((name, value) for name, value in dct.items() if not name.startswith('__'))
        uppercase_attr = dict((name.upper(), value) for name, value in attrs)
        return super().__new__(cls, name, bases, uppercase_attr)

class FooChild(metaclass=UpperAttrMetaclass):
    __metaclass__ = UpperAttrMetaclass
    bar = 'bilibili'
print('=======')
print(hasattr(FooChild, 'BAR'))

# 创建一个类似Django中的ORM来熟悉一下元类的使用，
# 通常元类用来创建API是非常好的选择，使用元类的编写很复杂但使用者可以非常简洁的调用API。
###########
# 元类使用

#一、首先来定义Field类，它负责保存数据库表的字段名和字段类型：
class Field(object):
    def __init__(self, name, column_type):
        self.name = name
        self.column_type = column_type
    def __str__(self):
        return '<%s: %s>' % (self.__class__, self.name)

class StringField(Field):
    def __init__(self, name):
        super().__init__(name, 'varchar(100)')

class IntegerField(Field):
    def __init__(self, name):
        super().__init__(name, 'bigint')

#二、定义元类，控制Model对象的创建
class ModelMetaclass(type):
    def __new__(cls, name, bases, attrs):
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        mappings = dict()
        for k, v in attrs.items():
            # 保存类属性和列的映射关系到mappings字典
            if isinstance(v, Field):
                print('Found mapping: %s==>%s' % (k, v))
                mappings[k] = v
        for k in mappings.keys():
            # 将类属性移除，使定义的类字段不污染User类属性，只在实例中可以访问这些key
            attrs.pop(k)
        attrs['__table__'] = name.lower()  # 假设表名和为类名的小写,创建类时添加一个__table__类属性
        attrs['__mappings__'] = mappings  # 保存属性和列的映射关系，创建类时添加一个__mappings__类属性
        return super().__new__(cls, name, bases, attrs)
#三、编写Model基类
class Model(dict, metaclass=ModelMetaclass):
    __metaclass__ = ModelMetaclass

    def __init__(self, **kw):
        super().__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Model' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value

    def save(self):
        fields = []
        params = []
        args = []
        for k, v in self.__mappings__.items():
            fields.append(v.name)
            params.append('?')
            args.append(getattr(self, k, None))
        sql = 'insert into %s (%s) values (%s)' % (self.__table__, ','.join(fields), ','.join(params))
        print('SQL: %s' % sql)
        print('ARGS: %s' % str(args))

#最后，我们使用定义好的ORM接口，使用起来非常的简单。
class User(Model):
    # 定义类的属性到列的映射：
    id = IntegerField('id')
    name = StringField('username')
    email = StringField('email')
    password = StringField('password')

# 创建一个实例：
u = User(id=12345, name='Michael', email='test@orm.org', password='my-pwd')
# 保存到数据库：
u.save()







