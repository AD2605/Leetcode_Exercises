#ifndef COMPTETITIVE_TREES_H
#define COMPTETITIVE_TREES_H
#include <bits/stdc++.h>
#include <pthread.h>
#include <atomic>

using namespace std;

typedef struct node{
    int data;
    struct node *left;
    struct node *right;
} node;

int getLevelUtil(struct node *node,
                 int data, int level)
{
    if (node == nullptr)
        return 0;

    if (node -> data == data)
        return level;

    int downlevel = getLevelUtil(node -> left,
                                 data, level + 1);
    if (downlevel != 0)
        return downlevel;

    downlevel = getLevelUtil(node->right,
                             data, level + 1);
    return downlevel;
}

int getLevel(struct node *node, int data)
{
    return getLevelUtil(node, data, 1);
}

node* newNode(int data){
    node* temp = new node;
    temp->data = data;
    temp->left = nullptr;
    temp->right = nullptr;
    return temp;
}

node* createTree(int num_level){
    auto root = newNode(rand());
    auto node = root;
    for(int i =0; i<num_level;++i){
        node->data = 23;
        node->right = newNode(rand());
        node->left = newNode(rand());
        node = node->right;
    }
    return root;
}

node* insertNode(node* root, int data){
    if(!root){
        root = newNode(rand());
        return root;
    }
    queue<node*> queue;
    queue.push(root);
    while (!queue.empty()){
     auto temp = queue.front();
     queue.pop();
     if(temp->left!= nullptr)
         queue.push(temp->left);
     else{
         temp->left = newNode(rand());
         return root;
     }
     if (temp->right!= nullptr)
         queue.push(temp->right);
     else{
         temp->right = newNode(rand());
         return root;
     }
    }
}

#endif //COMPTETITIVE_TREES_H


void postOrder(node* root){
    if(root == nullptr)
        return;
    postOrder(root->left);

    postOrder(root->right);

    cout<<root->data<<" ";
}

void inOrder(node* root){
    if(root == nullptr)
        return;
    
    inOrder(root->left);
    cout<<root->data<<" ";
    inOrder(root->right);
}

void preOrder(node* root){
    if(root==nullptr){
        return;
    }

    cout<<root->data<<" ";
    preOrder(root->left);
    preOrder(root->right);

}
