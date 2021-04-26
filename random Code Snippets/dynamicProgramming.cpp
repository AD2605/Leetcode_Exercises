#include <algorithm>
#include <cstdint>
#include <iostream>
#include <bits/stdc++.h>


std::vector<char> commonCharacters;

int max(int a, int b){
    return (a > b) ? a: b;
}

int min(int a, int b){
    return (a < b) ? a : b;
}

int LCS(char* x, char* y, int l1, int l2){
    if (l1 == 0 || l2 == 0)
        return 0;
    if(x[l1-1] == y[l2-1])
    {
        commonCharacters.emplace_back(x[l1-1]);
        return 1 + LCS(x, y, l1-1, l2-1);
    }
    else
        return max(LCS(x, y, l1, l2-1), LCS(x, y, l1-1, l2));
}

int LCSprint(char* x, char* y, int l1, int l2, std::vector<char> chars){
    if (l1 == 0 || l2 == 0)
    {
       reverse(chars.begin(), chars.end());
       for(int i=0; i < chars.size(); i++){
           std::cout<<chars[i];
       }
    }
    if(x[l1-1] == y[l2-1])
    {
        chars.emplace_back(x[l1-1]);
        return LCSprint(x, y, l1-1, l2-1, chars);
    }
    else 
    {
        int a = LCS(x, y, l1-1, l2);
        int b = LCS(x, y, l1, l2-1);

        a ? b = LCSprint(x, y, l1-1, l2, chars) : LCSprint(x, y, l1, l2-1, chars);
    }

    return 0;
}
template<typename type>
std::vector<type> LIS(std::vector<type> input){
    std::vector<type> lis;
    int num = 0;
    type current_num;
    for(int i=0; i< input.size()-1; i++){
        if(input[i] < input[i+1] and current_num < input[i]){
            current_num = input[i];
            lis.emplace_back(input[i]);
            num += 1;
        }
    }
    std::cout<<num<<std::endl;
    for (int j=0; j< lis.size(); j++){
        std::cout<<lis.at(j)<<" ";
    }
    std::cout<<std::endl;
    return lis;
}


int knapsack(std::vector<int> profits, std::vector<int> weights,  int capacity, int fractional=0){
    assert(profits.size() == weights.size());
    int n = profits.size();
    int table[n+1][capacity+1];

    std::map<int, int > mymap;
    for(int i=0; i<n; i++){
        mymap[weights.at(i)] = profits.at(i);
    }
    std::sort(weights.begin(), weights.end());

    for(int i = 0; i<=n; i++){
        int weight = -1;
        int profit = -1;
        if(i==0)
            ;
        else{
            weight = weights.at(i-1);
            profit = mymap[weights.at(i-1)];
        }
        for(int j=0; j<=capacity; j++){
            if (i == 0 || j == 0){
                table[i][j] = 0;
                continue;
            } 
            assert(profit != -1 && weight != -1);

            if (weight > j){
                table[i][j] = table[i-1][j];
            } 
            
            else{
                auto value = max(profit + table[i-1][j - weight], table[i-1][j]);
                table[i][j] = value;
            }
        }
    }
    int maxProfit = table[n][capacity];
    return maxProfit;
}


int minimumCoin(std::vector<int> coins, int value){
    int n = coins.size();
    int table[value + 1];
    table[0] = 0;
    for(int i=1; i<=value; i++){
        table[i] = INT_MAX;
    }

    for(int i =1; i<=value; i++){
        for(int j=0; j<n; j++){
            if(coins.at(j) <=i){
                int residual = table[i - coins.at(j)];
                if (residual != INT_MAX && residual + 1 < table[i])
                    table[i] = residual + 1;
            }
        }
    }

    return table[value];
}

int count( int S[], int m, int n )
{
    int i, j, x, y;
 
    // We need n+1 rows as the table
    // is constructed in bottom up
    // manner using the base case 0
    // value case (n = 0)
    int table[n + 1][m];
     
    // Fill the enteries for 0
    // value case (n = 0)
    for (i = 0; i < m; i++)
        table[0][i] = 1;
 
    // Fill rest of the table entries
    // in bottom up manner
    for (i = 1; i < n + 1; i++)
    {
        for (j = 0; j < m; j++)
        {
            // Count of solutions including S[j]
            x = (i-S[j] >= 0) ? table[i - S[j]][j] : 0;
 
            // Count of solutions excluding S[j]
            y = (j >= 1) ? table[i][j - 1] : 0;
 
            // total count
            table[i][j] = x + y;
        }
    }
    return table[n][m - 1];
}

int subsetSumExistence(std::vector<int> input, int target){
    int n = input.size();
    int table[n+1][target+1];
    int value;

    for(int j = 0; j<=n; j++)
        table[j][0] = 1;

    for(int i=0; i<=n; i++){
        if (i > 0){
            value = input.at(i -1);
        }
        for(int j=1; j<=target; j++){
            
            if(i == 0 && j != 0)
            {
                table[i][j] = 0;
                continue;
            }

            if (value > j)
            {
                table[i][j] = 0;
                continue;
            }
            else{
                table[i][j] = table[i][j - input.at(i-1)] || table[i-1][j];
            }
        }
    }
    return table[n][target];
}

int equalSumSplitExistence(std::vector<int> input, int target){
    int sum = 0;
    for(auto i : input){
        sum += i;
    }
    if ((sum % 2) != 0){
        return 0;
    }

    int existence = subsetSumExistence(input, target/2);
    return existence;
}

int countSubsetSum(std::vector<int> input, int target){
    int value;

     if(!subsetSumExistence(input, target)){
         return 0;
     }
     int n = input.size();
     int table[n+1][target+1];
     
    for(int j = 0; j<=n; j++)
         table[j][0] = 1;

        for(int i=0; i<=n; i++){
        if (i > 0){
            value = input.at(i -1);
        }
        for(int j=1; j<=target; j++){
            
            if(i == 0 && j != 0)
            {
                table[i][j] = 0;
                continue;
            }

            if (value > j)
            {
                table[i][j] = table[i-1][j];
                continue;
            }
            else{
                table[i][j] = table[i-1][j - input.at(i-1)] + table[i-1][j];
            }
        }
    }

    return table[n][target];
}

int minimumSubsetSumDifference(std::vector<int> input){
    int n = input.size();
      int sum = 0;
      for(int i=0; i<input.size(); i++){
          sum += input.at(i);
      }
      int target = sum;

          int table[n+1][target+1];
    int value;

    for(int j = 0; j<=n; j++)
        table[j][0] = 1;

    for(int i=0; i<=n; i++){
        if (i > 0){
            value = input.at(i -1);
        }
        for(int j=1; j<=target; j++){
            
            if(i == 0 && j != 0)
            {
                table[i][j] = 0;
                continue;
            }

            if (value > j)
            {
                table[i][j] = 0;
                continue;
            }
            else{
                table[i][j] = table[i][j - input.at(i-1)] || table[i-1][j];
            }
        }
    }
    std::vector<int> differences;
    int *last_row = *table + (target+1)*n;
    int minimum = INT_MAX;
    for(int i=0; i<=int(target/2); i++){
        if(*(last_row + i)){
            minimum = min(minimum, sum - 2*i);
        }
    }
    return minimum;
}

int subsetSumDifferenceCount(std::vector<int> input, int difference){
    int sum = 0;
    for(int i=0; i<input.size(); i++){
        sum += input.at(i);
    }
    
    int target = (difference + sum) / 2;
    int count = countSubsetSum(input, target);

    return count;
 }


int targetSum(std::vector<int> input, int target){
    int count = subsetSumDifferenceCount(input, target);
    return count;  
}


int unBoundedKnapSack(std::vector<int> weights, std::vector<int> profits, int capacity){

    assert(profits.size() == weights.size());
    int n = profits.size();
    int table[n+1][capacity+1];

    //std::sort(weights.begin(), weights.end());

    for (int i=0; i<=n; i++){
        int weight = -1;
        int profit = -1;

        if(i > 0){
            weight = weights.at(i-1);
            profit = profits.at(i-1);
        }

        for(int j=0;j<=capacity+1; j++){

            if(i == 0 || j == 0)
            {
                table[i][j] = 0; 
                continue;
            }

            assert(profit != -1 && weight != -1);

            if (weight > j){
                table[i][j] = table[i-1][j];
            } 
            
            else{
                auto value = max(profit + table[i][j - weight], table[i-1][j]);
                table[i][j] = value;
            }
        }
    }

    return table[n][capacity];
}


int rodCutting(int rod_length, std::vector<int> length, std::vector<int> prize){
    /*Given a rod cut it such that we maximize prize*/
    /*This is unbounded Kapsack*/
        assert(length.size() == prize.size());
    int n = length.size();
    int table[n+1][rod_length+1];


    //std::sort(weights.begin(), weights.end());

    for (int i=0; i<=n; i++){
        int weight = -1;
        int profit = -1;

        if(i > 0){
            weight = length.at(i-1);
            profit = prize.at(i-1);
        }

        for(int j=0;j<=rod_length+1; j++){

            if(i == 0 || j == 0)
            {
                table[i][j] = 0; 
                continue;
            }

            assert(profit != -1 && weight != -1);

            if (weight > j){
                table[i][j] = table[i-1][j];
            } 
            
            else{
                auto value = max(profit + table[i][j - weight], table[i-1][j]);
                table[i][j] = value;
            }
        }
    }

    return table[n][rod_length];
}

int numWaysCoinChange(std::vector<int> coins, int sum){
     
     int value;
     int n = coins.size();
     int table[n+1][sum+1];
     
    for(int j = 0; j<=n; j++)
         table[j][0] = 1;

        for(int i=0; i<=n; i++){
        if (i > 0){
            value = coins.at(i -1);
        }
        for(int j=1; j<=sum; j++){
            
            if(i == 0 && j != 0)
            {
                table[i][j] = 0;
                continue;
            }

            else if (value > j)
            {
                table[i][j] = table[i-1][j];
                continue;
            }
            else{
                table[i][j] = table[i][j - coins.at(i-1)] + table[i-1][j];
            }
        }
    }

    return table[n][sum];
}


int minimumNumberofCoinsChange(std::vector<int> coins, int target){
    if(target == 0)
        return 0;

    if(!subsetSumExistence(coins, target))
        return -1;
    
    int  n = coins.size();
    int table[n+1][target+1];

    for(int i = 0; i<=target; i++)
        table[0][i] = INT_MAX - 1;
     
    for(int i=0; i<=n; i++)
        table[i][0] = 0;

    for(int i=0; i<=target; i++){
        if( (i % coins.at(0)) == 0)
            table[1][i] = i / coins.at(0);
        else
            table[1][i] = INT_MAX - 1;
    }

    for(int i=2; i<=n; i++){
        int value = coins.at(i-1);
        for(int j=1; j<=target; j++){
            if(value > target)
                table[i][j] = table[i-1][j];
            else{
                table[i][j] = std::min(1 + table[i][j - value], table[i-1][j]);
            }
        }
    }
    return table[n][target];
}


int minJumpsRequired(std::vector<int> input){
    int n = input.size();
    int jumps[n];

    if(n == 0 || input.at(0) == 0)
        return INT_MAX;
    
    jumps[0] = 0;
    for(int i=1; i<n; i++){
        jumps[i] = INT_MAX;
        for(int j=0; j<i; j++){
            if(i <= j + input.at(j) && jumps[j] != INT_MAX){
                jumps[i] = std::min(jumps[i], jumps[j] + 1);
                break;
            }
        }
    }
    return jumps[n-1];
}

int main(){
    /*
    char x[] = "AGGTAB";
    char y[] = "GXTXAYB";
    int l1 = strlen(x);
    int l2 = strlen(y);
    auto chars  = std::vector<char>();

    int lcs = LCS(x, y, l1, l2);
    std::cout<<std::endl<<lcs<<std::endl;
    for(int i=0; i<commonCharacters.size(); i++){
        std::cout<<commonCharacters.at(i)<<"  ";
    }
    std::cout<<std::endl;
    */
    /*
    std::vector<int> weights{10, 20, 30};
    std::vector<int> profits{60, 100, 120};
    int capacity = 50;
    auto profit = knapsack(profits, weights, capacity);
    std::cout<<profit<<std::endl;*/
    /*
    std::vector<int> coins{1, 2, 5};
    int target = 11;
    std::cout<<minimumNumberofCoinsChange(coins, target)<<std::endl;
    */
    std::vector<int> input{1, 1, 0, 1, 0, 9};
    std::cout<<minJumpsRequired(input)<<std::endl;
}