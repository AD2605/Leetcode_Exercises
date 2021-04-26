#include <iostream>
#include <bits/stdc++.h>
#include <string>


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

int checkEndianess(){
    /*Returns 1 if little Endian Else big endian*/
    unsigned int i = 1;
    char *c = (char*) & i;
    if(*c)
        return 1;
    else 
        return 0;
}

int main(){
    std::vector<char> character = {'a', 'b', 'c', 'd'};
    auto subsets = allPossibleSubsets(character);
    std::cout<<checkEndianess()<<std::endl;
}