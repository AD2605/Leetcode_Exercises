class Solution:
    def myAtoi(self, str: str) -> int:
        try:
            num = int(str)
            if abs(num) > pow(2, 31) - 1 and num < 0:
                return -1 * pow(2, 31)
            elif abs(num) > pow(2, 31) - 1 and num > 0:
                return pow(2, 31) - 1
            else:
                return num
        except:
            try:
                number = ''
                flag = 0
                for letter in str:
                    if letter.isdigit() or letter == '-' or letter == ' ' or letter == '+':
                        if letter.isdigit():
                            flag = 1
                        if not letter.isdigit() and flag ==1:
                            break
                        number = number + letter
                    else:
                        break
                number = int(number)
                if abs(number) > pow(2, 31) - 1 and number < 0:
                    return -1 * pow(2, 31)
                elif abs(number) > pow(2, 31) - 1 and number > 0:
                    return pow(2, 31) - 1
                else:
                    return number
            except:
                return 0