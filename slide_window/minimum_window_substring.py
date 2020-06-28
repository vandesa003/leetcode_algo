"""
Leetcode No.76
Minimum Window Substring
这道题有点歧义，关于t的唯一性，我这份代码没有AC，但是思路没问题，只是关于t的唯一性的歧义部分有误。
思路依然是slide window,不过与之前2道不同的是，这个slide window需要start和end同时维护，移动一个，保持另一个不动。
有点类似于双指针了。
"""

from collections import defaultdict
from collections import Counter
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""
        if len(s) < len(t):
            return ""
        if s == t:
            return s
        st = 0
        en = 0
        table = {}
        min_window = len(s) + 1
        fn_s = 0
        fn_e = 0
        t_set = set()
        for i in range(len(t)):
            t_set.add(t[i])
        while en < len(s):
            if s[en] in t:
                table[s[en]] = table.get(s[en], 0) + 1
            
            if len(table) == len(t_set):
                while st <= en and len(table) == len(t_set):
                    if en - st + 1 < min_window:
                        min_window = en - st + 1
                        fn_s = st
                        fn_e = en
                    if s[st] in t:
                        table[s[st]] -= 1
                        if table[s[st]] == 0:
                            table.pop(s[st])
                    st += 1
            en += 1
        if min_window == len(s) + 1:
            return ""
        else:
            return s[fn_s:fn_e+1]