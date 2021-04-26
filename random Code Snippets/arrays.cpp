#include <iostream>
#include <bits/stdc++.h>

using namespace std;

int min(int a, int b){
    return a < b ? a : b;
}

int max(int a, int b){
    return a > b? a : b;
}

int maxsumSubarray(std::vector<int> input){
    // Kadane Algorithm
    int n = input.size();
    int max_so_far = INT_MIN;
    int max_ending_here = 0;

    for(int i=0; i<n; i++){
        max_ending_here = max_ending_here + input.at(i);
        if(max_so_far < max_ending_here)
            max_so_far = max_ending_here;
        
        if (max_ending_here < 0 )
            max_ending_here = 0;
    }

    return max_so_far;
}

int maxProductSubarray(std::vector<int> input){
    int n = input.size();
    int max_ending_here = 1;
    int min_ending_here = 1;
    int max_so_far = 0;
    int flag = 0;

    for(int i=0; i<n; i++){
        if(input.at(i) > 0){
            max_ending_here = max_ending_here * input.at(i);
            min_ending_here = min(min_ending_here*input.at(i), 1);
            flag = 1;
        }
        else if (input.at(i) == 0){
            max_ending_here = 1;
            min_ending_here = 1;
        }
        else{
            int temp = max_ending_here;
            max_ending_here = max(min_ending_here * input.at(i), 1);
            min_ending_here = temp * input.at(i);
        }
        if(max_so_far < max_ending_here)
            max_so_far = max_ending_here;
    }
    if (flag == 0 && max_so_far == 0)
        return 0;

    return max_so_far;
}

std::pair<int, int> pairsumClosest(std::vector<int> input, int x){
    
    int n = input.size();
    int index1, index2; 
    int l = 0;
    int r = n - 1;
    int difference = INT_MAX;

    while(r > 1){
        if(abs(input.at(l) + input.at(r) - x) < difference){
            index1 = l;
            index2 = r;
            difference = abs(input.at(l) + input.at(r) - x);
        }
        if(input.at(l) + input.at(r)> x)
            r--;
        else
            l++;
    }

    std::pair<int, int>result(input.at(index1), input.at(index2));
    return result;

}

int *st;
  

int findGcd(int ss, int se, int qs, int qe, int si)
{
    if (ss>qe || se < qs)
        return 0;
    if (qs<=ss && qe>=se)
        return st[si];
    int mid = ss+(se-ss)/2;
    return __gcd(findGcd(ss, mid, qs, qe, si*2+1),
               findGcd(mid+1, se, qs, qe, si*2+2));
}
  

int findRangeGcd(int ss, int se, int arr[],int n)
{
    if (ss<0 || se > n-1 || ss>se)
    {
        cout << "Invalid Arguments" << "\n";
        return -1;
    }
    return findGcd(0, n-1, ss, se, 0);
}
  
int constructST(int arr[], int ss, int se, int si)
{
    if (ss==se)
    {
        st[si] = arr[ss];
        return st[si];
    }
    int mid = ss+(se-ss)/2;
    st[si] = __gcd(constructST(arr, ss, mid, si*2+1),
                 constructST(arr, mid+1, se, si*2+2));
    return st[si];
}

int *constructSegmentTree(int arr[], int n)
{
   int height = (int)(ceil(log2(n)));
   int size = 2*(int)pow(2, height)-1;
   st = new int[size];
   constructST(arr, 0, n-1, 0);
   return st;
}

void rightrotate(int arr[], int n, int outofplace, int cur)
{
    char tmp = arr[cur];
    for (int i = cur; i > outofplace; i--)
        arr[i] = arr[i - 1];
    arr[outofplace] = tmp;
}
 
void negativePositiveAlternate(int arr[], int n)
{
    int outofplace = -1;
 
    for (int index = 0; index < n; index++)
    {
        if (outofplace >= 0)
        {

            if (((arr[index] >= 0) && (arr[outofplace] < 0))
                || ((arr[index] < 0)
                    && (arr[outofplace] >= 0)))
            {
                rightrotate(arr, n, outofplace, index);
 
                // the new out-of-place entry is now 2 steps
                // ahead
                if (index - outofplace >= 2)
                    outofplace = outofplace + 2;
                else
                    outofplace = -1;
            }
        }

        if (outofplace == -1) {
            if (((arr[index] >= 0) && (!(index & 0x01)))
                || ((arr[index] < 0) && (index & 0x01))) {
                outofplace = index;
            }
        }
    }
}

#define MAX 500
 
int multiply(int x, int res[], int res_size);
 
// This function finds factorial of large numbers
// and prints them
void factorial(int n)
{
    int res[MAX];
 
    // Initialize result
    res[0] = 1;
    int res_size = 1;
 
    // Apply simple factorial formula n! = 1 * 2 * 3 * 4...*n
    for (int x=2; x<=n; x++)
        res_size = multiply(x, res, res_size);
 
    cout << "Factorial of given number is \n";
    for (int i=res_size-1; i>=0; i--)
        cout << res[i];
}
 
// This function multiplies x with the number
// represented by res[].
// res_size is size of res[] or number of digits in the
// number represented by res[]. This function uses simple
// school mathematics for multiplication.
// This function may value of res_size and returns the
// new value of res_size
int multiply(int x, int res[], int res_size)
{
    int carry = 0;  // Initialize carry
 
    // One by one multiply n with individual digits of res[]
    for (int i=0; i<res_size; i++)
    {
        int prod = res[i] * x + carry;
 
        // Store last digit of 'prod' in res[] 
        res[i] = prod % 10; 
 
        // Put rest in carry
        carry  = prod/10;   
    }
 
    // Put carry in res and increase result size
    while (carry)
    {
        res[res_size] = carry%10;
        carry = carry/10;
        res_size++;
    }
    return res_size;
}


int main(){
    std::vector<int> input{1, -2, -3, 0, 7, -8, -2 };
    std::cout<<maxsumSubarray(input)<<std::endl;
    std::cout<<maxProductSubarray(input)<<std::endl;
    return 0;
}