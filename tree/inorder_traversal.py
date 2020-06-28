"""
中序遍历：DFS或者栈来实现。
leetcode No.94
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # dfs 递归实现
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        ans = []
        def dfs(node):
            if not node:
                return node
            dfs(node.left)
            ans.append(node.val)
            dfs(node.right)
        dfs(root)
        return ans

    # dfs 栈实现(非递归)
