class Solution:
    def decodeAtIndex(self, S: str, K: int) -> str:
        tape = ''
        for char in S:
            if char.isalpha():
                tape = tape+char
            if char.isnumeric():
                tape = tape+ (int(char)-1)*tape
        return tape[K-1]
    