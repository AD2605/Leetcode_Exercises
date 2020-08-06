from math import ceil, floor
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        
        result = (dividend/divisor)
        
        if (result) > pow(2, 31) -1 or result < pow(-2, 31):
            return pow(2, 31) - 1
        if result<0:
            return(ceil(result))
        
        if result >= 0:
            return(int(result))