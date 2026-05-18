
print(ord('牛'))
chr(0)
print(f"-{chr(0)}-")
print("this is a test" + chr(1) + "string")

s = "this is a test" + chr(0) + "string"

print(repr(s))
# 'this is a test\x00string'

print(s)
# this is a teststring

test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(list(utf8_encoded)
      )

# requires `regex` package
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
red = re.findall(PAT, "some text that i'll pre-tokenize")
print(red)
red = re.finditer(PAT,"some text that i'll pre-tokenize")
print(red)
print(max([("A", "B"), ("BCA", "C"), ("BC", "A"), ("BA", "B")]))