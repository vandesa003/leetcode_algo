"""
quick sort in python.
quick sort is all about partition, and it's a recursive implementation.

快排的核心就是分治，某种程度上来说也是一种二分思想。
Created On 25th June, 2020
Author: bohang.li
"""
import random


def partition(arr, low, high):
    i = low - 1  # smaller index
    pivot = random.randint(low, high)
    arr[pivot], arr[high] = arr[high], arr[pivot]
    pivot = arr[high]
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    # the pivot value hasn't processed!
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1


def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)


if __name__ == "__main__":
    data = [10, 7, 8, 9, 1, 5]
    n = len(data)
    quick_sort(data, 0, n-1)
    print(data)
