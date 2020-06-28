"""
Leetcode No.3 
Longest Substring Without Repeating Characters

method: slide window
time complexity: O(n)
space complexity: O(n)
"""

from collections import Counter
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        i = 0
        j = 0
        n = len(s)
        table = {}
        ans = 0
        while j<n and i<n:
            if table.get(s[j], None) == None or (table.get(s[j], None)<i or table.get(s[j], None)>j):
                table[s[j]] = j
                if (j - i + 1) > ans:
                    ans = j - i + 1
            else:
                i = table[s[j]] + 1
                table[s[j]] = j
            j += 1
            # print(i, j)
            
        return ans
