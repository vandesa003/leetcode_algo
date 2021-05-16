"""
Leetcode 24.
Swap Nodes in Pairs.

很简单的链表题目，注意技巧，使用多余的root node来方便查找链表头。
24ms beats 96%
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        root = ListNode()
        root.next = head
        mem = root
        cur = root.next
        change = False
        while cur:
            if change:
                tmp = cur.next
                cur.next = pre
                pre.next = tmp
                mem.next = cur
                cur = cur.next
                mem = cur
            else:
                pre = cur
            cur = cur.next
            change = not change
        return root.next
