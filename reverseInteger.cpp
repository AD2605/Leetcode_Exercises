#include <string>

using namespace std;

class Solution {
public:
    int reverse(int x) {
        int reversed = 0;
        if (x <0) {
            string str = to_string(x);
            string rev = string(str.rbegin(), str.rend());
            reversed = -1*stoi(rev);
        }
        if(x>0){
            string str = to_string(x);
            string rev = string(str.rbegin(), str.rend());
            reversed = stoi(rev);
        }
        return reversed;
    }
};