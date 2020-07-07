"""
Leetcode 120.
Triangle

比较典型的DP，第一道二维DP！
被二维list的生成方式坑了：
千万不要用：
[[-1]*m]*n 这种方式生成二维list，因为这样生成的是浅拷贝。
正确生成方式为：
[[-1] * m for _ in range(m)]

时间复杂度：O(n^2)
空间复杂度：O(n^2)
"""

# 递归DP
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = [[-1] * len(triangle) for _ in range(len(triangle))]
        if not triangle or len(triangle) == 0:
            return 0
        def dfs(row, col):
            if row == 0 and col == 0:
                dp[row][col] = triangle[0][0]
                return triangle[0][0]

            if dp[row][col] == -1:
                if row > col > 0:
                    dp[row][col] = min(dfs(row-1, col-1), dfs(row-1, col)) + triangle[row][col]
                elif col == 0:
                    dp[row][col] = dfs(row-1, col) + triangle[row][col]
                else:
                    dp[row][col] = dfs(row-1, col-1) + triangle[row][col]
                return dp[row][col]
            else:
                return dp[row][col]
        ans = float("inf")
        for i in range(len(triangle)):
            tmp = dfs(len(triangle)-1, i)
            if tmp < ans:
                ans = tmp
        return ans


# 非递归DP
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle) == 1:
            return triangle[0][0]
        dp = [[0] * len(triangle) for i in range(len(triangle))]
        dp[0][0] = triangle[0][0]
        ans = float("inf")
        if not triangle or len(triangle) == 0:
            return 0
        for row in range(1, len(triangle)):
            for col in range(len(triangle[row])):
                if col == 0:
                    dp[row][col] = dp[row-1][col] + triangle[row][col]
                elif col == len(triangle[row])-1:
                    dp[row][col] = dp[row-1][col-1] + triangle[row][col]
                else:
                    dp[row][col] = min(dp[row-1][col-1], dp[row-1][col]) + triangle[row][col]
                if row == len(triangle) - 1:
                    if dp[row][col] < ans:
                        ans = dp[row][col]
        return ans
