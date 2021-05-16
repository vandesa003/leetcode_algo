"""
Leetcode 53.
Maximum Subarray

DP 入门题目

解法一：双指针法
用一个辅助序列sums保存前n个元素之和。
问题变成了：
如果当前子序列之和小于0，则抛弃之前的所有序列，以当前节点为起点，重新计算，最后记录最大的子序列和即可。
时间复杂度：O(n)
没觉着这个是DP的题目，可能解法还是有问题。

解法二：DP
dp[i] = max(nums[i], dp[i-1]+nums[i])
时间复杂度：O(n)
"""


# 解法一： 双指针 120ms, Beat 14%
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums or len(nums) == 0:
            return 0
        sums = [0] * (len(nums) + 1)
        sums[0] = 0
        max_sum = float("-inf")
        for i in range(1, len(nums)+1):
            sums[i] = sums[i-1] + nums[i-1]
        st = 0
        for i in range(1, len(sums)):
            pre_sum = sums[i] - sums[st]
            if pre_sum <= 0:
                st = i
            if pre_sum > max_sum:
                max_sum = pre_sum
        return max_sum


# 解法二： DP 68ms, Beat 76%
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(nums[i], dp[i-1]+nums[i])
        res = max(dp)
        return res
