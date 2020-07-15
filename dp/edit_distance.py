"""
Leetcode 72.
Edit Distance.

2D DP problem. 经典！
"""
from collections import defaultdict

class Solution_origin:
    def minDistance(self, word1: str, word2: str) -> int:
        
        # 表示word1[:i] 到 word2[:j]的最小编辑距离
        def dfs(i, j):
            if i == -1:
                return j + 1
            if j == -1:
                return i + 1
            if word1[i] == word2[j]:
                return dfs(i-1, j-1)
            else:
                delete = dfs(i-1, j) + 1  # 删除word1中第i元素，或者也可以理解为在word2中插入第j+1元素。二者等价，但是我们得确定一个参考系，这里选word1为参考。注意操作次数+1。
                insert = dfs(i, j-1) + 1  # 在word1中插入第j元素。
                replace = dfs(i-1, j-1) + 1  # 将word1中的第i元素替换为word2中第j元素。那么此时2个元素相等，指针同时往前走。
                return min(insert, delete, replace)
        return dfs(len(word1)-1, len(word2)-1)


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = {}
        # 表示word1[:i] 到 word2[:j]的最小编辑距离
        def dfs(i, j):
            if (i,j) in dp:
                return dp[(i,j)]
            if i == -1:
                dp[(i,j)] = j+1
                return dp[(i,j)]
            if j == -1:
                dp[i,j] = i+1
                return dp[(i,j)]
            if word1[i] == word2[j]:
                dp[(i,j)] = dfs(i-1, j-1)
            else:
                delete = dfs(i-1, j) + 1  # 删除word1中第i元素，或者也可以理解为在word2中插入第j+1元素。二者等价，但是我们得确定一个参考系，这里选word1为参考。注意操作次数+1。
                insert = dfs(i, j-1) + 1  # 在word1中插入第j元素。
                replace = dfs(i-1, j-1) + 1  # 将word1中的第i元素替换为word2中第j元素。那么此时2个元素相等，指针同时往前走。
                dp[(i,j)] = min(insert, delete, replace)
            return dp[(i,j)]
        res = dfs(len(word1)-1, len(word2)-1)
        return res


if __name__ == "__main__":
    s1 = "dinitrophenylhydrazine"
    s2 = "benzalphenylhydrazone"
    sol = Solution()
    ans = sol.minDistance(s1, s2)
    print(ans)
