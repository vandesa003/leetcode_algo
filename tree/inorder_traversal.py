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
def inorderTraversal_stack(root):
    if not root:
        return root
    stack = []
    ans = []
    while len(stack)>0 or root:
        # 先遍历完所有左子树
        if root is not None:
            stack.append(root)
            root = root.left
        # 左子树遍历完后，弹出父节点，遍历右子树
        else:
            root = stack.pop()
            ans.append(root.val)
            root = root.right
    return ans
