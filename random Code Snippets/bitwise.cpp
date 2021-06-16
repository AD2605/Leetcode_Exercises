#include <cstdint>
#include <iostream>
#include <bits/stdc++.h>
#include <string>

#define mysize(object) (char *)(&object+1) - (char *)(&object)


int getSingle(std::vector<int> input){
    /*
    Works only if each number occurs for an odd number of time*/
    int ones = 0;
    int twos = 0;
    int common_bit_mask;

    for(int i=0; i<input.size(); i++){
        twos = twos | (ones & input.at(i));
        ones = ones ^ input.at(i);
        common_bit_mask = ~(ones & twos);
        ones &= common_bit_mask;
        ones &= common_bit_mask;
        twos &= common_bit_mask;
    }
    return ones;
}

int getOppositeSign(int x, int y){
    return ((x ^ y) < 0);
}


int countBitstobeFlipped(int a, int b){
    int count = 0;
    auto n = a ^b;
    while(n > 0){
        count++;
        n &= (n-1);
    }
    return count;
}

int turnoffRightmost(unsigned int n){
    return n & (n-1);
}

int swapNibbles(int x){
    return ((x & 0x0F) << 4 | (x & 0xF0) >> 4);
}


std::vector<std::string> allPossibleSubsets(std::vector<char> characters){
    std::vector<std::string> Subsets;

    for(int i = 0; i< (1 << characters.size()); i++){
        std::string subsetString = "";
        for(int j=0; j<characters.size(); j++){
            if (i & (1 << j))
                subsetString += characters.at(j);
        }
        Subsets.emplace_back(subsetString);
    }

    std::sort(Subsets.begin(), Subsets.end());
    return Subsets;
}

template<typename type>
type russianPeasant(type a, type b){
    type result = 0;
    while(b > 0){
        if (b & 1)
            result = result + a;
        a = a<<1;
        b = b>>1;
    }
    return result;
}

int getIthBit(int n, int i){
    return n & (1<<i);
}

int setIthBit(int n, int i){
    return n | (1<<i);
}

int clearIthBit(int n, int i){
    return n & (~(1<<i));
}

int toggle_Ith_Bit(int n, int i){
    return (n ^ (1 << (i-1)));
}

int checkEndianess(){
    /*Returns 1 if little Endian Else big endian*/
    unsigned int i = 1;
    char *c = (char*) & i;
    if(*c)
        return 1;
    else 
        return 0;
}

char* toggleString(char* input){
    for (int i=0; *(input + i)!='\0'; i++){
        *(input + i) ^= 32;
    }
    return input;
}

template<typename type>
type additionUsingBitwise(type a, type b){

    while(b != 0){
        type carry = a & b;
        a = a ^ b;
        b = carry << 1;
    }

    return a;
}

void ReLU(float* input, size_t size){
    for (size_t i=0; i<size; i++){
        * (input + i) = ((std::int32_t(*(input + i))>>31) + 1.0f) * *(input + i);
    }
}

int min(int x, int y) 
{ 
    return y ^ ((x ^ y) & -(x < y)); 
} 

/*Function to find maximum of x and y*/
int max(int x, int y) 
{ 
    return x ^ ((x ^ y) & -(x < y));  
} 



int main(){
    /*
    char str[] = "AtHaRvA DuBeY";
    //char a = 'a';
    std::vector<char> character = {'a', 'b', 'c', 'd'};
    auto subsets = allPossibleSubsets(character);
    std::cout<<checkEndianess()<<std::endl;
    std::cout<<toggleString(str)<<std::endl;
    std::cout<<additionUsingBitwise<int>(20, -25)<<std::endl;
    std::cout<<char(additionUsingBitwise(int('a'), 62))<<std::endl;
    std::cout<<russianPeasant<int>(5000, 155323)<<std::endl;
    */

    float a = 10.0f;
    std::cout<<sizeof(a)<<std::endl;
    std::cout<<mysize(a)<<std::endl;
}
