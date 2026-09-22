from pytoolbox import DataStandardizer, chunked, clamp, is_palindrome

print("clamp:", clamp(120, 0, 100))
print("palindrome:", is_palindrome("Never odd or even"))
print("chunks:", list(chunked([1, 2, 3, 4, 5], 2)))

standardizer = DataStandardizer()
standardizer.fit([10, 20, 30])
print("standardized:", standardizer.transform([10, 20, 30]))
