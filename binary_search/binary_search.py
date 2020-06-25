"""
Basic Binary Search For python.

Created On 25th June, 2020
Author: bohang.li
"""

# 精准的二分查找
def binary_search_recursive(arr, low, high, value):
    
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] > value:
            return binary_search_recursive(arr, low, mid - 1, value)
        elif arr[mid] < value:
            return binary_search_recursive(arr, mid + 1, high, value)
        else:
            return mid
    else:
        return -1


# 精准的二分查找
def binary_search_iterative(arr, value):
    low = 0
    high = len(arr) - 1
    while(low < high):
        mid = (low + high) // 2
        if arr[mid] > value:
            high = mid - 1
        elif arr[mid] < value:
            low = mid + 1
        else:
            return mid
    return -1


# lower bound, 大于等于查找
def lower_bound(arr, value):
    """
    find the index of the first element in arr >= value.
    """
    low = 0
    high = len(arr) - 1
    while(low < high):
        mid = (low + high) // 2
        if arr[mid] < value:
            low = mid + 1
        else:
            found = mid
            high = mid - 1
    return found
    

# upper bound, 小于等于查找
def upper_bound(arr, value):
    """
    find the index of the last element in arr <= value.
    """
    low = 0
    high = len(arr) - 1
    while(low < high):
        mid = (low + high) // 2
        if arr[mid] > value:
            high = mid - 1
        else:
            found = mid
            low = mid + 1
    return found


if __name__ == "__main__":
    data = [1, 3, 4, 5, 6, 9 ,10 ,32, 92]
    print(binary_search_recursive(data, 0, len(data), 92))
    print(binary_search_iterative(data, 3))
    print(lower_bound(data, 2))
    print(upper_bound(data, 33))
