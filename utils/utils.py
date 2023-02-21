def has_int_sqrt(n):
    return int(n**0.5) == n**0.5

class magicNestedDict:
    def __init__(self, d):
        for k, v in d.items():
            if type(v) == dict:
                self.__setattr__(k, magicNestedDict(v))
            else:
                self.__setattr__(k, v)