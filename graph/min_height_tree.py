"""
Leetcode 310.
Minimum Height Trees
逐层删除叶子节点，最后剩下的1-2个节点即是所需。
"""

from collections import defaultdict
from collections import deque


class Solution:
    def findMinHeightTrees(self, n: int, edges: list) -> list:
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]
        graph = defaultdict(set)
        for i in range(len(edges)):
            graph[edges[i][0]].add(edges[i][1])
            graph[edges[i][1]].add(edges[i][0])
        while len(graph) > 2:
            leaves = []
            for i in graph:
                if len(graph[i]) == 1:
                    leaves.append(i)
            for i in range(len(leaves)):
                leaf = leaves[i]
                neighbors = graph[leaf]
                for nei in neighbors:
                    graph[nei].remove(leaf)
                graph.pop(leaf)
        return list(graph.keys())
        

if __name__ == "__main__":
    n = 6
    edges = [[0,1],[0,2],[0,3],[3,4],[4,5]]
    sol = Solution()
    print(sol.findMinHeightTrees(n, edges))
