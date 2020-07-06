"""
Leetcode 70.
Climbing Stairs.

DP.
类似斐波那契数列：
转移方程： f(n) = f(n-1) + f(n-2).
时间复杂度：O(n)
还是没看明白这跟DP有啥关系，就是递归而已。
"""

class Solution:
    def climbStairs(self, n: int) -> int:
        res = [-1] * (n)
        def dfs(n):
            if n == 1:
                return 1
            if n == 2:
                return 2
            if res[n-1] == -1:
                res[n-1] = dfs(n-1) + dfs(n-2)
                return res[n-1]
            else:
                return res[n-1]
        ans = dfs(n)
        return ans
