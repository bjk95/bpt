from typing import Self


class Value:
    def __init__(self, data: float, _children: tuple[Self, Self]=(), _operation: str = '') -> None:
        self.data: float = data
        self.prev = set(_children)
        self.operation = _operation
    
    def __repr__(self) -> str:
        return f'Value({self.data})'
    
    def __add__(self, other: Self):
        return Value(self.data + other.data, (self,other), '+')
    
    def __mul__(self, other: Self):
        return Value(self.data * other.data, (self,other), "*")
    
a = Value(10)
b = Value(15)
c = Value(3)
print(a + b * c)