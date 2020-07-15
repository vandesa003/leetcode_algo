"""
Leetcode 322.
Coin Change.

经典DP问题。
本题使用dfs来做。
dp[i]表示amount为i时最小兑换次数。但是多加一个状态，当dp[i]=-1时，说明无法兑换，例如i=11， coins=[2]的时候。
我们知道，贪心地拿最大面额并不一定能保证最小兑换次数，例如i=8, coins=[5,4,1]的时候，先拿5，需要换4次，而先拿4，只需要换2次。
所以还是需要每个面额都试一下。
dp[i] = dp[i-coins[j]] + 1
时间复杂度: 不好计算。amount要根据coins减到负数，然后再循环coins个数次。
"""


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = {}
        coins.sort()  # O(nlogn)
        num = bisect.bisect_left(coins, amount)
        def dfs(cur_amount):
            if cur_amount == 0:
                if cur_amount not in dp:
                    dp[cur_amount] = 0
                return dp[cur_amount]
            if cur_amount < 0:
                if -1 not in dp:
                    dp[-1] = -1
                return dp[-1]
            if cur_amount not in dp:
                tmp = float("inf")
                for i in range(len(coins)):
                    if dfs(cur_amount-coins[i]) != -1 and dfs(cur_amount-coins[i]) < tmp:
                        tmp = dfs(cur_amount-coins[i]) + 1
                    
                if tmp != float("inf"):
                    dp[cur_amount] = tmp
                else:
                    dp[cur_amount] = -1
            return dp[cur_amount]
        res = dfs(amount)
        return res
