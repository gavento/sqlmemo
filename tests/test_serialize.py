import dataclasses
import math
import unittest
from sqlcache.serialize import hash_obj, jsonize


@dataclasses.dataclass
class Person:
    name: str
    age: int


@dataclasses.dataclass
class Book:
    title: str
    author: Person


class JsonizeTest(unittest.TestCase):

    def test_primitive_types(self):
        self.assertEqual(jsonize(42), 42)
        self.assertEqual(jsonize(3.14), 3.14)
        self.assertEqual(jsonize(True), True)
        self.assertEqual(jsonize("hello"), "hello")
        self.assertEqual(jsonize(None), None)
        self.assertEqual(jsonize([1, "two", 3.0]), [1, "two", 3.0])
        self.assertEqual(jsonize((1, "two", 3.0)), [1, "two", 3.0])
        self.assertEqual(jsonize(set()), [])
        self.assertEqual(set(jsonize({1, "two", 3.0})), {1, "two", 3.0})
        self.assertEqual(jsonize({"a": 1, "b": 2, "c": 3}), {"a": 1, "b": 2, "c": 3})
        self.assertEqual(jsonize({"a": [1, 2, 3], "b": {"x": 1, "y": 2}}), {"a": [1, 2, 3], "b": {"x": 1, "y": 2}})

    def test_dataclass(self):
        person = Person(name="Alice", age=25)
        self.assertEqual(jsonize(person), {"name": "Alice", "age": 25})

        book = Book(title="My Book", author=person)
        self.assertEqual(jsonize(book), {"title": "My Book", "author": {"name": "Alice", "age": 25}})


class HashObjTest(unittest.TestCase):

    def test_primitive_types(self):
        # Test hashing of primitive types
        self.assertEqual(hash_obj(42), hash_obj(42))
        self.assertEqual(hash_obj(3.14), hash_obj(3.14))
        self.assertEqual(hash_obj(math.nan), hash_obj(math.nan))
        self.assertEqual(hash_obj(True), hash_obj(True))
        self.assertEqual(hash_obj("hello"), hash_obj("hello"))
        self.assertEqual(hash_obj(None), hash_obj(None))

        # Test hashing of lists
        self.assertEqual(hash_obj([1, "two", 3.0]), hash_obj([1, "two", 3.0]))
        self.assertNotEqual(hash_obj([1, "two", 3.0]), hash_obj([1, 3.0, "two"]))

        # Test hashing of tuples
        self.assertEqual(hash_obj((1, "two", 3.0)), hash_obj((1, "two", 3.0)))
        self.assertNotEqual(hash_obj((1, "two", 3.0)), hash_obj((1, 3.0, "two")))

        # Test hashing of sets
        self.assertEqual(hash_obj(set()), hash_obj(set()))
        self.assertNotEqual(hash_obj(set()), hash_obj({1, 2, 3}))

        # Test hashing of dictionaries
        self.assertEqual(hash_obj({"a": 1, "b": 2, "c": 3}), hash_obj({"a": 1, "b": 2, "c": 3}))
        self.assertNotEqual(hash_obj({"a": 1, "b": 2, "c": 3}), hash_obj({"a": 1, "b": 2}))

    def test_dataclass(self):
        # Test hashing of dataclasses
        person = Person(name="Alice", age=25)
        self.assertEqual(hash_obj(person), hash_obj(person))
        self.assertNotEqual(hash_obj(person), hash_obj(Person(name="Bob", age=30)))

        book = Book(title="My Book", author=person)
        self.assertEqual(hash_obj(book), hash_obj(book))
        self.assertNotEqual(hash_obj(book), hash_obj(Book(title="Other Book", author=person)))

    def test_stability(self):
        self.assertEqual(hash_obj(42), "c74908b0bb857be44562081a7cd89cad00144c15bcda4732f3b64d7b0b058f46")
        self.assertEqual(hash_obj(3.14), "0ee07a0f8cf3414e22ef3bf186a2b0de6e470f922ef91cd47e0d6154794801ff")
        self.assertEqual(hash_obj(math.nan), "55abd65ab1bec0ad930a0c080f2386d1c9c1598a0f28229eb462af99843d3f77")
        self.assertEqual(hash_obj(None), "85200570164e25d639a78faffa46800f88c755face8cbb1fc8a3bb7f8aaa1fca")
        self.assertEqual(
            hash_obj((["a"], {1, 2}, {1})),
            "2c9a38b2535ab4d885ee8f64f6504c3ba7273fc4e12159dfa7dca88b68de01e3",
        )
        self.assertEqual(
            hash_obj(Book(title="My Book", author=Person(name="Alice", age=25))),
            "cb40e87e83567459e4f2d8ec087e5cbf32918798184cfd1633e5a58c6b12fb8b",
        )


if __name__ == "__main__":
    unittest.main()
