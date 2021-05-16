"""
Leetcode 100.
Same Tree.

简单的树的遍历，用dfs遍历比较即可。
时间复杂度：O(n)，n为树中的节点数量，28ms，Beat 89%
"""


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def dfs(t1, t2):
            if t1 is None and t2 is None:
                return True
            if (t1 is None and t2 is not None) or (t1 is not None and t2 is None):
                return False
            if t1.val != t2.val:
                return False
            return dfs(t1.left, t2.left) and dfs(t1.right, t2.right)
        return dfs(p,q)
