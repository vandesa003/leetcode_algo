"""
Leetcode 15
3 sums.

这类问题类比2-sums，解法时间复杂度为O(n^2)
可以在2-sums的前面再加一层循环，也可以使用2 pointers，
从而不用hashmap来做存储，在时间复杂度都为O(n^2)的情况下，
优化空间复杂度。总之，这道题耗了我很长时间，后来发现，还是
two-pointers香！

这个题目的难点在于如何避免重复元素，以及减少不必要的循环。
由于我们知道时间复杂度已经O(n^2)了，所以O(nlogn)的排序，
我们也可以大胆使用了。
所以秘诀就在，提前给一个有序数列，利用有序数列避免重复。
"""

from collections import defaultdict
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = set()
        if not nums:
            return []
        if len(nums) < 3:
            return []
        nums.sort()
        for i in range(len(nums)-2):
            l = i + 1
            r = len(nums) - 1
            if i > 0 and nums[i] == nums[i-1]:
                continue
            while l < r:
                if nums[i] + nums[l] + nums[r] < 0:
                    l += 1
                elif nums[i] + nums[l] + nums[r] > 0:
                    r -= 1
                else:
                    res.add((nums[i], nums[l], nums[r]))
                    l += 1
        return map(list, res)
