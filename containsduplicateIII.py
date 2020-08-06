class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        nums.sort()
        l = len(nums)
        if t> len(nums):
            if abs(nums[l-1] - nums[0]< k):
                return True
            else:
                return False
        if t< len(nums):
            diff = abs(nums[l-1] - nums[l-1-t])
            if diff < k:
                return True
            else:
                return False