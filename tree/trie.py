"""
leetcode No.208
字典树，用于快速的字符串查找
"""


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = dict()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        p = self.root
        for x in word:
            if x not in p:
                p[x] = dict()
            p = p[x]
        p[-1] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        p = self.root
        for i in range(len(word)):
            next_node = p.get(word[i], None)
            if not next_node:
                return False
            p = next_node
        return True if p.get(-1, None) else False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        p = self.root
        for i in range(len(prefix)):
            next_node = p.get(prefix[i], None)
            if not next_node:
                return False
            p = next_node
        return True
