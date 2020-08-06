class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split(" ")
        words = list(filter(lambda a: a!='', words))
        new_S = ''
        for word in reversed(words):
            new_S = new_S + word
            new_S = new_S + ' '
        return new_S[:-1]