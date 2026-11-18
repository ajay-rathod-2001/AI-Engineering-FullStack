from typing import TypedDict
class Person(TypedDict):
    name: str
    age: int
    
new_persons: Person = {'name':'Agentic', 'age': 20}

print(new_persons)