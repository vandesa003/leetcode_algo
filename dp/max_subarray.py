"""
Leetcode 53.
Maximum Subarray

DP 入门，用一个辅助序列sums保存前n个元素之和。
问题变成了：
如果当前子序列之和小于0，则抛弃之前的所有序列，以当前节点为起点，重新计算，最后记录最大的子序列和即可。
时间复杂度：O(n)
没觉着这个是DP的题目，可能解法还是有问题。
"""

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
