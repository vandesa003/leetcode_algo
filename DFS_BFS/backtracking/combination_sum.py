"""
Leetcode 39. 
Combination Sum

经典回溯。DFS，递归即可。
"""

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        def dfs(n, can_set, cur_ans):
            if n == 0:
                ans.append(cur_ans)
            if n < 0:
                return
            else:
                for i in range(len(can_set)):
                    dfs(n-can_set[i], can_set[i:], cur_ans+[can_set[i]])
        dfs(target, candidates, [])
        return ans
