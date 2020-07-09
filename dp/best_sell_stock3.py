"""
Leetcode 123.
Best Time to Buy and Sell Stock III.

3d-dp. hard.
https://labuladong.gitbook.io/algo/dong-tai-gui-hua-xi-lie/tuan-mie-gu-piao-wen-ti
"""


class Solution:
    def maxProfit(self, prices: list) -> int:
        if len(prices) < 2:
            return 0
        dp = [[[float("-inf")] * len(prices) for x in range(3)] for y in range(2)]
        dp[0][2][0] = 0  # 未持有股票，可买2次，第0天
        dp[1][1][0] = -prices[0]  # 持有股票，可买1次，第0天
        for i in range(1, len(prices)):
            for j in range(3):
                dp[0][j][i] = max(dp[0][j][i-1], dp[1][j][i-1] + prices[i])  # 未持有股票，可买j次，第i天
                if j != 2:
                    dp[1][j][i] = max(dp[0][j+1][i-1] - prices[i], dp[1][j][i-1])  # 持有股票，可买j次，第i天
        ans = max(dp[0][0][len(prices)-1], dp[0][1][len(prices)-1], dp[0][2][len(prices)-1])
        if ans > 0:
            return ans
        else:
            return 0


if __name__ == "__main__":
    prices = [4,2,1,7]
    sol = Solution()
    ans = sol.maxProfit(prices)
    print(ans)
