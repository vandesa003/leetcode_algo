"""
Basic Binary Search For python.

Created On 25th June, 2020
Author: bohang.li
"""

def binary_search_recursive(arr, low, high, value):
    
    if high >= low:
        mid = (high + low) // 2
        if arr[mid] > value:
            return binary_search_recursive(arr, low, mid, value)
        elif arr[mid] < value:
            return binary_search_recursive(arr, mid, high, value)
        else:
            return mid
    else:
        return -1


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


if __name__ == "__main__":
    data = [1, 3, 4, 5, 6, 9 ,10 ,32, 92]
    print(binary_search_recursive(data, 0, len(data), 4))
    print(binary_search_iterative(data, 4))
