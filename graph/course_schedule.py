"""
Leetcode 207
Course Schedule.
拓扑排序，DFS，BFS。
"""

# DFS实现，有点难理解
# 查看图中是不是有环，或者说，这道题的本质就是判断一个图是否为DAG。
# 如果DFS过程中，访问到了being_visited中的节点，则说明图中有环。
# https://leetcode.com/problems/course-schedule/discuss/658297/Python-Topological-sort-with-recurcive-dfs-explained
class Solution1:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        from collections import defaultdict
        graph = defaultdict(list)
        # 记录
        visited = set()
        # 建图
        for x, y in prerequisites:
            graph[y].append(x)

        # 深度遍历
        def dfs(i, being_visited):
            if i in being_visited:
                return False
            if i in visited:
                return True
            visited.add(i)
            being_visited.add(i)
            for j in graph[i]:
                if not dfs(j, being_visited):
                    return False
            being_visited.remove(i)
            return True
        # 检测每门功课起始是否存在环
        for i in range(numCourses):
            # 已经访问过
            if i in visited: continue
            if not dfs(i, set()): return False
        return True


# BFS实现，比较容易理解，但是比DFS略慢。
# 从入度为0的节点开始查找，每次把当前的节点(入度为0)从图中删除。
# 最后判断图中节点是否为空，不为空则说明图中有环。
class Solution2:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        degree = [0] * numCourses
        # 建图
        for x, y in prerequisites:
            graph[y].append(x)
            degree[x] += 1
        queue = deque([i for i in range(numCourses) if degree[i] == 0])
        #print(queue)
        while queue:
            i = queue.pop()
            for j in graph[i]:
                degree[j] -= 1
                if degree[j] == 0:
                    queue.appendleft(j)
            graph.pop(i)
        return len(graph) == 0
