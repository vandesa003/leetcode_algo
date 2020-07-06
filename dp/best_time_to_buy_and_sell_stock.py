"""
Leetcode 121.
Best Time to Buy and Sell Stock

这题终于有点DP的意思了，依然是一道一维DP。用一个DP数组记录第i步的最佳选择：
最后返回DP数组的最后一个元素即可。
"""

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices or len(prices) < 2:
            return 0
        dp = [0] * len(prices)
        min_p = prices[0]
        for i in range(len(prices)):
            dp[i] = max(dp[i-1], prices[i] - min_p)
            min_p = min(min_p, prices[i])
        return dp[-1]
