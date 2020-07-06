"""
Leetcode 210.
Course Schedule II

Topological Sort / BFS / DFS.
"""

from collections import defaultdict
from collections import deque
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        degree = defaultdict(int)
        frontier = deque()
        sort_list = []
        for i in range(len(prerequisites)):
            graph[prerequisites[i][1]].append(prerequisites[i][0])
            degree[prerequisites[i][0]] += 1
        for i in range(numCourses):
            if degree[i] == 0:
                frontier.append(i)
        while len(frontier) > 0:
            cur = frontier.popleft()
            sort_list.append(cur)
            for nei in graph[cur]:
                degree[nei] -= 1
                if degree[nei] == 0:
                    frontier.append(nei)
            graph.pop(cur)
        if len(graph) != 0:
            return []
        else:
            return sort_list
