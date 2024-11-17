import numpy as np
import pandas as pd
import timeit
from collections import defaultdict

# Approach 1: Using defaultdict
def group_totals_defaultdict(strArr: list[str]) -> str:
    """
    Aggregates values by key using a defaultdict approach.

    Args:
        strArr (list[str]): List of strings in the format "key:value".

    Returns:
        str: A string of aggregated key-value pairs, sorted by key.
    """
    data = defaultdict(int)
    for item in strArr:
        # Check if the item contains exactly one colon and splits into two parts
        if ":" in item and len(item.split(":")) == 2:
            key, value = item.split(":")
            try:
                # Aggregate the values by key
                data[key] += int(value)
            except ValueError:
                # Skip items where the value cannot be converted to an integer
                continue
    # Return the aggregated key-value pairs as a sorted string
    return ",".join(f"{key}:{value}" for key, value in sorted(data.items()))

# Approach 2: Using NumPy
def group_totals_numpy(strArr: list[str]) -> str:
    """
    Aggregates values by key using a NumPy approach.

    Args:
        strArr (list[str]): List of strings in the format "key:value".

    Returns:
        str: A string of aggregated key-value pairs.
    """
    # Filter valid items
    valid_items = [s for s in strArr if s.count(":") == 1 and len(s.split(":")) == 2]
    if not valid_items:
        return ""
    # Split and convert values
    keys, values = zip(*[s.split(":") for s in valid_items])
    values = np.array(list(map(int, values)))
    keys = np.array(keys)
    # Aggregate values by unique keys
    unique_keys = np.unique(keys)
    summed_values = np.array([values[keys == key].sum() for key in unique_keys])
    # Return the aggregated key-value pairs as a string
    return ",".join(f"{key}:{value}" for key, value in zip(unique_keys, summed_values))

# Approach 3: Using pandas
def group_totals_pandas(strArr: list[str]) -> str:
    """
    Aggregates values by key using a pandas DataFrame.

    Args:
        strArr (list[str]): List of strings in the format "key:value".

    Returns:
        str: A string of aggregated key-value pairs.
    """
    # Filter valid items and create DataFrame
    valid_items = [s.split(":") for s in strArr if ":" in s and len(s.split(":")) == 2]
    if not valid_items:
        return ""
    df = pd.DataFrame(valid_items, columns=["key", "value"])
    # Convert 'value' column to integers, handling errors
    df["value"] = pd.to_numeric(df["value"], errors='coerce').fillna(0).astype(int)

    # Group by 'key' and sum the 'value'
    grouped_result = df.groupby("key")["value"].sum()
    # Return the aggregated key-value pairs as a string
    return ",".join(f"{key}:{value}" for key, value in grouped_result.items())

# Generate large test data
n = 50000  # Number of elements
keys = np.random.choice(['A', 'B', 'C', 'D'], size=n)
values = np.random.randint(-100, 100, size=n)
strArr = [f"{key}:{value}" for key, value in zip(keys, values)]

# Measure performance
time_defaultdict = timeit.timeit(lambda: group_totals_defaultdict(strArr), number=10)
time_numpy = timeit.timeit(lambda: group_totals_numpy(strArr), number=10)
time_pandas = timeit.timeit(lambda: group_totals_pandas(strArr), number=10)

# Print the performance results
print(f"Performance Comparison (10 runs):")
print(f"Defaultdict Approach: {time_defaultdict:.4f} seconds")
print(f"NumPy Approach: {time_numpy:.4f} seconds")
print(f"Pandas Approach: {time_pandas:.4f} seconds")