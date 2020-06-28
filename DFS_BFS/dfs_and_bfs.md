## DFS and BFS

曾经以为DFS和BFS很高大上，当时实习的时候被一个参加过ACM的小伙子一顿DFS，BFS给我忽悠瘸了。直到如今，我终于有个空闲的时间，不玩游戏不约妹纸也不偷懒，好好地研究一番才发现，这玩意居然是graph里最简单最基本的算法。可以理解为graph里的暴力搜索，因为BFS和DFS都是遍历图结构中的每一条边和每一个节点，因此他们的时间复杂度都是O(V+E)，V是节点，E是边。但是不同的是，BFS的空间复杂度更高，形象的理解是因为BFS看得更远，而DFS则是埋头苦干，不撞南墙不回头，因此BFS心里装的事情更多，而DFS则是干就完了，轻装上阵，赢了会所嫩模，输了下海干活。

讲的严谨一点则是，DFS是要用栈来实现的，通常就是递归，遍历过的点，就会被pop出去，内存里只会存储当前的路径。BFS则是用队列来实现的，通常就是python里的list，BFS每走一步，都要查看下一步的所有可能性，所以自然需要存储的节点就更多，内存消耗也更大。一图说明：

DFS：只保存最后的路径的信息：

![img](https://miro.medium.com/max/1080/1*jv9MI87yGVSASgnqEKfQGQ.png)

BFS：保存几乎一大半的graph信息：

![img](https://miro.medium.com/max/1080/1*r2vmFbUB1Cl6g3CCuw22Og.png)



图片来源：https://medium.com/tebs-lab/breadth-first-search-and-depth-first-search-4310f3bf8416

当然这里的graph包括了我们通常理解的无向图、有向图、有向循环图、有向无环图(DAGs)等，也包括了二叉树等树结构。

总之，BFS和DFS给了我们一个基础范式，就是对于图结构来说，我们应该如何实现节点遍历(暴力搜索)。虽然他们的效果就是暴力搜索的效果，但是这两个方法也是两个非常重要的思维方式，因为我发现，在了解他们之前，给我一张图，我甚至不知道如何去暴力搜索！

接下来写个伪代码实现：

BFS(Queue)：

```
BFS(graph, start_node, end_node):
    frontier = new Queue()
    frontier.enqueue(start_node)
    explored = new Set()
    
    while frontier is not empty:
        current_node = frontier.dequeue()
        if current_node in explored:
            continue
        if current_node == end_node:
            return success
        for neighbor in graph.get_neighbors(current_node):
            frontier.enqueue(neighbor)
        explored.add(current_node)
```

DFS(Stack):

```
DFS(graph, start_node, end_node):
    frontier = new Stack()
    frontier.push(start_node)
    explored = new Set()
    
    while frontier is not empty:
        current_node = frontier.pop()
        if current_node in explored:
            continue
        if current_node == end_node:
            return success
        for neighbor in graph.get_neighbors(current_node):
            frontier.push(neighbor)
        explored.add(current_node)
```

only difference is queue and stack.

注意一点：在python中，队列请不要使用list，而是使用`collections.deque`因为list的pop方法是O(n)复杂度，而正常队列的pop方法应该是O(1)复杂度。

### BFS

今天完成了BFS专题里的6道题目。https://leetcode.com/tag/breadth-first-search/

3道easy，3道medium。主要涉及到了二叉树的层级遍历，基本上熟悉了队列的写法，注意在python里不要用list作为队列。

### DFS

今天完成了2到medium的DFS，本质上，上面的blog已经讲得非常清楚了，就是栈的数据结构的应用，或者说是递归。