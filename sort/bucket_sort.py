from collections import defaultdict
from typing import List

"""
Leetcode 912.
input range: [-50000, 50000], int 
"""
def bucket_sort(arr: List[int]) -> List:
    bucket = defaultdict(int)
    res = []
    for i in range(len(arr)):
        bucket[arr[i]] += 1
    for i in range(-50000, 50001):
        res.extend(bucket[i]*[i])
    return res