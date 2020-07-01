Leetcode All In One: https://github.com/grandyang/leetcode

基础算法 —— 代码模板链接 常用代码模板1——基础算法

排序
二分
高精度
前缀和与差分
双指针算法

回朔法

位运算
离散化
区间合并
数据结构 —— 代码模板链接 常用代码模板2——数据结构

链表与邻接表：树与图的存储
栈与队列：单调队列、单调栈
kmp
Trie
并查集
堆
Hash表
C++ STL使用技巧
搜索与图论 —— 代码模板链接 常用代码模板3——搜索与图论

DFS与BFS
树与图的遍历：拓扑排序
最短路
最小生成树
二分图：染色法、匈牙利算法
数学知识 —— 代码模板链接 常用代码模板4——数学知识

质数
约数
欧拉函数
快速幂
扩展欧几里得算法
中国剩余定理
高斯消元
组合计数
容斥原理
简单博弈论
动态规划

背包问题
线性DP
区间DP
计数类DP
数位统计DP
状态压缩DP
树形DP
记忆化搜索
贪心

时空复杂度分析



提高知识点
动态规划——从集合角度考虑DP问题

1.1 数字三角形模型
1.2 最长上升子序列模型
1.3 背包模型
1.4 状态机模型
1.5 状态压缩DP
1.6 区间DP
1.7 树形DP
1.8 数位DP
1.9 单调队列优化的DP问题
1.10 斜率优化的DP问题
搜索

2.1 BFS
2.1.1 Flood Fill
2.1.2 最短路模型
2.1.3 多源BFS
2.1.4 最小步数模型
2.1.5 双端队列广搜
2.1.6 双向广搜
2.1.7 A*
2.2 DFS
2.2.1 连通性模型
2.2.2 搜索顺序
2.2.3 剪枝与优化
2.2.4 迭代加深
2.2.5 双向DFS
2.2.6 IDA*
图论

3.1.1 单源最短路的建图方式
3.1.2 单源最短路的综合应用
3.1.3 单源最短路的扩展应用
3.2 floyd算法及其变形
3.3.1 最小生成树的典型应用
3.3.2 最小生成树的扩展应用
3.4 SPFA求负环
3.5 差分约束
3.6 最近公共祖先
3.7 有向图的强连通分量
3.8 无向图的双连通分量
3.9 二分图
3.10 欧拉回路和欧拉路径
3.11 拓扑排序
高级数据结构

4.1 并查集
4.2 树状数组
4.3.1 线段树（一）
4.3.2 线段树（二）
4.4 可持久化数据结构
4.5 平衡树——Treap
4.6 AC自动机
数学知识

5.1 筛质数
5.2 分解质因数
5.3 快速幂
5.4 约数个数
5.5 欧拉函数
5.6 同余
5.7 矩阵乘法
5.8 组合计数
5.9 高斯消元
5.10 容斥原理
5.11 概率与数学期望
5.12 博弈论
基础算法

6.1 位运算
6.2 递归
6.3 前缀和与差分
6.4 二分
6.5 排序
6.6 RMQ



## Binary Search

2020.06.25

今天的目标，就是熟悉二分法的思路和写法。

二分法的思路很简单。小时候有个综艺节目，猜家电的价格，如果能够准确猜对，就把相应的家电送给参与者。譬如一台电视机价格为2000块。

我们第一次猜：3000。

主持人：高了。

那我们第二次肯定不会猜2999，否则要想猜到正确的价格，需要尝试1000次！

既然高了，我们又知道价格肯定在0-3000块之间。

第二次就折中猜一个：1500块。

主持人这回又说：低了。

现在我们就知道了，这个价格在1500-3000之间。

第三次我们再折中猜：2250块。

主持人：高了。

1500-2250.

第四次：1875。低了。

1875-2250.

第五次：2062。高了。

1875-2062。

第六次：1968。低了。

1968-2062。

第七次：2015。高了。

1968-2015。

第八次：1991。低了。

1991-2015。

第九次：2003。高了。

1991-2003。

第十次：1997。低了。

1997-2003。

第十一次：2000。答对了！！！

二分的威力，把2000次的搜索空间降到了11次。这也就是时间复杂度中log(n)的来历了。每次都砍掉当前搜索空间的一半，可不就是把原本为n的搜索空间变成了log(n)吗？！

二分的核心思想，很简单，但是难就难在，怎么去实现，或者说，把实际问题巧妙的转化为二分法。

那么现在，我们也清楚了，与二分搜索(binary search)对应的是线性搜索(linear search)。线性搜索对应的时间复杂度就是O(n)。



## Quik Sort

快排的本质就是分治算法，也是一种二分思想。

首先，找到一个支点(pivot)，然后用一个双指针相向地检索arr中的元素，将小于pivot的值放在左边，大于pivot的值放在右边，最后，将pivot移至中间。就得到了2个子序列(partition)，左边的partition都是小于pivot的，右边的都是大于pivot的。当然，两个partition内部还不是有序的。

![quick_sort_partition_animation](https://www.tutorialspoint.com/data_structures_algorithms/images/quick_sort_partition_animation.gif)

由于每个partition内部都还不是有序的，那么，接下来只需要对每个partition再重复上述的操作，直到所有的partition都不可分，排序完成。



快排的步骤：
```
Step 1 − Make the right-most index value pivot(最右端index的值作为pivot，也可以选别的)
Step 2 − partition the array using pivot value(根据pivot将arr分组)
Step 3 − quicksort left partition recursively(递归左边分组)
Step 4 − quicksort right partition recursively(递归右边分组)
```



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



## Tree

最近有个感悟，二分法就是算法里最牛的思想，为什么这么说？在最开始的Binary Search中我就说过，二分法是将搜索空间指数级降低的方法。可是日常生活中，并不是所有的数据结构或者问题都可以用二分法来解决。所以本质上，二叉树这种数据结构就是为了将二分这种方法运用到淋漓尽致。

参考刷题：https://zhuanlan.zhihu.com/p/57929515

114，105， 106，108，109，589，94，173，145，104

### preorder traversal

二叉树的先序遍历： 根左右，在我看来是最正常的一种二叉树遍历方式，可以用DFS、BFS来实现。



### inorder traversal

二叉树的中序遍历：左根右

leetcode 94

### postorder traversal

二叉树的后序遍历：左右根

### level order traversal 

二叉树的层序遍历。这个是符合人眼观察的遍历顺序。一般用BFS来实现。

### BST

binary search tree.

### Trie

字典树，或者叫前缀查询树。用于word查询，每个字母是树上的一个节点。可以用于搜索时的自动补全等功能。

实际上是一种数据结构，而非算法。一般包含如下方法：

1. 查询：简单，就是hashmap+树的查询。
2. 添加：简单，就是树的节点添加。
3. 删除：困难，如果要删除一个非叶子节点，需要考虑补全其他已经存在的节点。
4. startwith：简单，天生就是干这个的。



## Slide window

严格意义上来说，slide window更像是一种技术，而非算法。一般用于字符串处理，对于一个连续的字符串或者链表，求一些最大不重复子串或者最大相同子串。这些问题都有一个核心共性：单向的滑动窗口就可以解决问题。

一般来说，这种字符串类型的题目，如果是暴力搜索需要$$O(n^2)$$或者$$O(n^3)$$的复杂度。但是通过slide window配合hashmap来让算法记住一些迭代过程中的关键信息，就可以将复杂度优化到$$O(n)$$。

目前这部分的题目，我做了2道，一道medium一道hard。总结下来2点最重要，第一，设计好hashmap该存哪些信息，怎么存。第二，设置好滑窗的步长，划窗的一个特性是end节点每次循环后，都要+1，这是保证$$O(n)$$的关键，而start节点则可以根据所求解的问题来自定步长。

来一套模板吧：

```python
def slide_window(s:list):
  	ans = []
    table = {}
    start_point = 0
    end_point = 0
    while end_point < len(s):  # 以end_point为主循环
      	table.update  # 更新hashmap的内容
        if condition: 
          	table.update  # 更新hashmap的内容（如果有必要的话）。
            start_point = xxxx  # 更新start_point
        end_point += 1
    return ans  # 返回结果
```



## Union Find

并查集。用于2-D数组的连通域数量，大小等问题的求解。

其实这类问题也可以用BFS或者DFS来解决，并且理论上的时间复杂度也是O(M*N)(M和N为2D数组的长宽)。但是并查集是一种更好的数据结构，或者简单来说，BFS或者DFS在实现的过程中，如果对Hashmap管理足够好的话，其实就是实现了并查集的功能。

OK，接下来来看看并查集这个数据结构长啥样：

并查集的实现，只需要实现3个函数：

init：首先，初始化父节点list，让每个节点的父节点都是自己。

find：输入一个节点，查询该节点的根节点。这里要注意，根节点的判断依据是：该节点的父节点是否为自己。查询所需要的时间复杂度为树深。

union：当满足一定条件时(依题目而定)，将输入的2个节点合并。也就是将输入的2个节点的根节点合并。首先用find方法分别查询2个节点的根节点，其次，根据2个节点到根节点的深度。更深的作为父节点，另一个作为子节点。这样就合并了2个节点。当然，这个合并方法不是一成不变的，也可以单纯的把第一个节点的根节点作为2个节点的共同根节点。不过第一种方法考虑了树深，更优化一点而已。

并查集总体来说也就是一个O(n)的方法，理论上来说，没有比BFS或者DFS优越到哪里去。。。不过算是掌握了一种新方法吧。

## 拓扑排序



## 最小生成树

