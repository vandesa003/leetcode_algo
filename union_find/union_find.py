"""
template for union find data structure.

Created On 1st July 2020
Author: bohang.li
"""

class UnionFind(object):
    def __init__(self, grid:list):
        self.parents = self.init_parents(grid)

    def init_parents(self, grid):
        parents = []
        n, m = len(grid), len(grid[0])
        for i in range(n):
            for j in range(m):
                parents.append(i*m+j)
        return parents

    def find(self, x:int):
        # 查找root节点，当x不是root节点时，不断向上查，直到查到root。root节点的特征是root的父节点就是自己。
        while self.parents[x] != x:
            self.parents[x] = self.parents[self.parents[x]]
            x = self.parents[x]
        return x

    def union(self, x, y):
        # 当满足一定条件时(通常是根据题意)，我们将x和y节点合并。首先分别找到x和y的父节点，当二者父节点不同时，
        # 令y的父节点为x的父节点。这里可以根据树的平衡性来优化，比如判断此时的树深，更深的树作为父节点。这样
        # 可以减少查询所需的次数，从而提高效率。
        x_parent = self.find(x)
        y_parent = self.find(y)
        if x_parent != y_parent:
            self.parents[y_parent] = x_parent
