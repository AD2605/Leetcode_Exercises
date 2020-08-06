class Solution:
    def isNumber(self, s: str) -> bool:
        try:
            num = float(s)
            return True
        except:
            return False
        