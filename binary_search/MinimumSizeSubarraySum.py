"""
209. Minimum Size Subarray Sum(Medium)

思路1：序列求和，可以将无序数列转换成单调递增序列，从而使用二分减少搜索空间。
时间复杂度： O(n*log(n))

思路2: 双指针法/滑窗法(2-pointer/slide-window),用一段窗口过一遍序列即可。
时间复杂度： 接近O(n)，比O(n)略大。
"""
def lower_bound(arr, value):
    """
    find the index of the first element in arr >= value.
    """
    low = 0
    high = len(arr) - 1
    found = len(arr)
    while(low <= high):
        mid = (low + high) // 2
        if arr[mid] < value:
            low = mid + 1
        else:
            found = mid
            high = mid - 1
    return found


def minSubArrayLen_binary_search(s: int, nums: list) -> int:
    n = len(nums)
    ans = n
    if n == 0 or sum(nums) < s:
        return 0
    summ = [0]*(n+1)
    for i in range(1, n+1):
        summ[i] = summ[i-1] + nums[i-1]
    print(summ)
    for i in range(1, n):
        to_find = s + summ[i-1]
        ix = lower_bound(summ, to_find)
        print(i, ix)
        if ix != n + 1:
            ans = min(ans, ix - i + 1)
    return ans


def minSubArrayLen_slide_window(s: int, nums: list) -> int:
    n = len(nums)
    ans = n + 1
    summ = 0
    left = 0
    for i in range(n):
        summ += nums[i]
        while summ >= s:
            summ -= nums[left]
            ans = min(ans, i - left + 1)
            left += 1
    if ans == n + 1:
        return 0
    else:
        return ans


if __name__ == "__main__":
    data = [1,2,3,4,5]
    val = 15
    print(minSubArrayLen_binary_search(val, data))
    print(minSubArrayLen_slide_window(val, data))
