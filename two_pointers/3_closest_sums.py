"""
Leetcode 16
3 sum closest.

这类问题类比2-sums，解法时间复杂度为O(n^2)
可以在2-sums的前面再加一层循环，也可以使用2 pointers，
从而不用hashmap来做存储，在时间复杂度都为O(n^2)的情况下，
优化空间复杂度。总之，这道题耗了我很长时间，后来发现，还是
two-pointers香！
"""

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if not nums or len(nums) < 3:
            return None
        min_margin = float("inf")
        res = 0
        nums.sort()
        for i in range(len(nums) - 2):
            l = i + 1
            r = len(nums) - 1
            while l < r:
                tmp = target - (nums[l] + nums[r] + nums[i])
                cur_margin = abs(tmp)
                if cur_margin < min_margin:
                    min_margin = cur_margin
                    res = (nums[l] + nums[r] + nums[i])
                if tmp < 0:
                    r -= 1
                else:
                    l += 1
        return res
