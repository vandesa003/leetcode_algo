"""
Leetcode 55.
Jump Game.

Taged as greedy, but I currently put it in backtracking.

For this solution, it's a simple recursive/dfs
"""

class Solution:
    def canJump(self, nums: list) -> bool:
        arr = nums
        dp = {}
        def dfs(jump_from, jump_to):
            if jump_from in dp:
                return dp[jump_from]
            else:
                if jump_from == jump_to:
                    dp[jump_from] = True
                    return dp[jump_from]
                if jump_from > jump_to:
                    dp[jump_from] = False
                    return dp[jump_from]
                if arr[jump_from] < 1:
                    dp[jump_from] = False
                    return dp[jump_from]
                for i in range(1, arr[jump_from]+1):
                    if dfs(jump_from + i, jump_to) is True:
                        dp[jump_from] = True
                        return dp[jump_from]
                dp[jump_from] = False
                return dp[jump_from]
        res = dfs(0, len(nums)-1)
        return res


class Solution:
    def canJump(self, nums: list) -> bool:
        last_pos = len(nums) - 1
        for i in range(len(nums)-1, -1, -1):
            if i + nums[i] >= last_pos:
                last_pos = i
        return last_pos == 0


if __name__ == "__main__":
    case = [2,0,6,9,8,4,5,0,8,9,1,2,9,6,8,8,0,6,3,1,2,2,1,2,6,5,3,1,2,2,6,4,2,4,3,0,0,0,3,8,2,4,0,1,2,0,1,4,6,5,8,0,7,9,3,4,6,6,5,8,9,3,4,3,7,0,4,9,0,9,8,4,3,0,7,7,1,9,1,9,4,9,0,1,9,5,7,7,1,5,8,2,8,2,6,8,2,2,7,5,1,7,9,6]
    print(len(case))
    sol = Solution()
    ans = sol.canJump(case)
    print(ans)
