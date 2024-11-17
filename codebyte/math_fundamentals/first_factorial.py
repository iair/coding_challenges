# Have the function FirstFactorial(num) take the num parameter being passed and return the factorial of it. 
# For example: if num = 4, then your program should return (4 * 3 * 2 * 1) = 24. 
# For the test cases, the range will be between 1 and 18 and the input will always be an integer.

import math
import timeit
from functools import reduce

# Iterative approach
def first_factorial_1(num: int) -> int:
    """
    Calculate the factorial of a given non-negative integer using an iterative approach.
    """
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")
    result: int = 1
    for i in range(1, num + 1):
        result *= i
    return result

# Optimized approach using `reduce()`
def first_factorial_2(num: int) -> int:
    """
    Calculate the factorial of a given positive integer using the `reduce()` function.
    """
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")
    return reduce(lambda x, y: x * y, range(1, num + 1), 1)

# Built-in approach using `math.factorial()`
def first_factorial_3(num: int) -> int:
    """
    Calculate the factorial of a given non-negative integer using the built-in `math.factorial()`.
    """
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")
    return math.factorial(num)

# Test range: 1 to 18 (as specified in the problem)
test_cases = list(range(1, 19))

# Measure performance for all three functions
iterative_times = timeit.timeit(
    stmt="[first_factorial_1(n) for n in test_cases]",
    globals=globals(),
    number=100000
)

reduce_times = timeit.timeit(
    stmt="[first_factorial_2(n) for n in test_cases]",
    globals=globals(),
    number=100000
)

math_times = timeit.timeit(
    stmt="[first_factorial_3(n) for n in test_cases]",
    globals=globals(),
    number=100000
)

# Print the performance comparison
print("Performance Comparison:")
print(f"Iterative Approach (first_factorial_1): {iterative_times:.6f} seconds")
print(f"Reduce Approach (first_factorial_2): {reduce_times:.6f} seconds")
print(f"Math Approach (first_factorial_3): {math_times:.6f} seconds")

# Performance Comparison:
# Iterative Approach (first_factorial_1): 0.471636 seconds
# Reduce Approach (first_factorial_2): 0.827730 seconds
# Math Approach (first_factorial_3): 0.085048 seconds


