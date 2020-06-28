"""
Leetcode No.424
Longest Repeating Character Replacement

method: slide window
time complexity: O(n)
space complexity: O(n)
"""

from collections import defaultdict
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        st = 0
        en = 0
        table = defaultdict(int)
        max_freq = 0
        max_len = 0
        while en < len(s):
            table[s[en]] += 1
            max_freq = max(max_freq, table[s[en]])
            if en - st + 1 - max_freq > k:
                table[s[st]] -= 1
                st += 1
            else:
                max_len = max(max_len, en - st + 1)
            en += 1
        return max_len
