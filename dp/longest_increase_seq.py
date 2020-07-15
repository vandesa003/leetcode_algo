"""
Leetcode No. 300
Longest Increasing Subsequence

最长递增子序列，非常经典的一道题。
提供3种做法：
1. 递归 dfs DP 复杂度O(n^2) 这道题非递归更快，应该是递归实现有点问题，之前的题目都是递归更快。。。
2. 非递归 DP 复杂度O(n^2)
3. 二分 DP 复杂度O(nlogn)

DP的思路：
dp[i] 表示以nums[i]结尾的最长递增子序列长度，注意一定以nums[i]结尾，也就是nums[i]为子序列中最大元素。
dp[i] = 在nums[0:i]找到比nums[i]小的dp[i]中的最长子序列长度，再+1即可。

二分可优化的思路：
可以看到，普通的DP思路是O(n^2)的复杂度，但是里面有一步找num[0:i]中比nums[i]小的元素，想到什么了呢？
对，我循环找需要O(n)的复杂度，但是如果用二分找，只需要O(logn)的复杂度，这不就可以提升效率了？
继续想，我们在第一个循环里其实在不断地计算新的dp[i]，然后再第二层循环里对已知的dp元素进行一些查找工作，
那么有没有可能，我们在计算dp[i]元素的时候，就把dp组织成一个方便查找的数据结构？
平衡二叉树！或者堆！还是难。。。过几天再来看。
"""

# recursive 3704ms beat 5%
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_len = 0
        dp = [-1 for i in range(len(nums))]
        def dfs(i):
            if dp[i] != -1:
                return dp[i]
            if i == 0:
                if dp[i] == -1:
                    dp[i] = 1
                return dp[i]
            tmp = 0
            for j in range(i):
                cur = dfs(j)
                if nums[j] < nums[i]:
                    if cur > tmp:
                        tmp = dfs(j)
            dp[i] = tmp + 1
            return dp[i]
        dfs(len(nums)-1)
        return max(dp)

# dp table 1180ms beat 50%
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_len = 0
        dp = [-1 for i in range(len(nums))]
        dp[0] = 1
        for i in range(1, len(nums)):
            tmp = 0
            for j in range(i):
                if nums[j] < nums[i] and dp[j] > tmp:
                    tmp = dp[j]
            dp[i] = tmp + 1
        return max(dp)
