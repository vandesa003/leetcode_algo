## Quick Sort
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

