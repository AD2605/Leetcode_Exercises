#include<iostream>
#include <list>
#include<memory>
#include<algorithm>
#include<atomic>
#include<iostream>
#include<climits>
#include<bits/stdc++.h>

using namespace std;


class Graph{
public:
	Graph(int num_vertices){
		this->vertices = num_vertices;
		this->adjascency = new list<int>[this->vertices];
	}

	void insertEdgeDirected(int v1, int v2){
		this->adjascency[v1].push_back(v2);
	}

	void insertEdgeUndirected(int v1, int v2){
		this->adjascency[v1].push_back(v2);
		this->adjascency[v2].push_back(v1);
	}

	int cycleDetectorUndirected(int v, int visited[], int parent){
		visited[v] = 1;
		for(auto i = this->adjascency[v].begin(); i != this->adjascency[v].end(); ++i){
			if(!visited[*i]){
				if(cycleDetectorUndirected(*i, visited, v))
					return 1;
			}
			else if(*i != parent)
				return 1;
		}
		return 0;
	}


	int cycleDetectorDirected(int v, int visited [], int recursionStack[]){
		if(!visited[v]){
			visited[v] = 1;
			recursionStack[v] = 1;
			for(auto i = this->adjascency[v].begin(); i != this->adjascency[v].end(); i++){
				if(!visited[*i] && this->cycleDetectorDirected(*i, visited, recursionStack))
					return 1;
				else if(recursionStack[*i])
					return 1;
			}
		}
		recursionStack[v] = 0;
		return 0;
	}

	int isCyclic(int isDirected=1){
		if(isDirected){
		auto visited = new int[this->vertices];
		auto recursionStack = new int[this->vertices];
		for(int i=0; i<this->vertices; i++){
			visited[i] = 0; 
			recursionStack[i] = 0;
		}
		for(int i=0; i<this->vertices; i++){
			if(this->cycleDetectorDirected(i, visited, recursionStack))
				return 1;
		}
		return 0;
		}
		else{
			auto visited = new int[this->vertices];
			for(int i=0; i<this->vertices; i++)
				visited[i] = 0;

			for(int i=0; i<this->vertices; i++){
				if(!visited[i]){
					if(cycleDetectorUndirected(i, visited, -1))
						return 1;
				}
			}
			return 0;
		}
	}

private:
	std::list<int> * adjascency;
	int vertices;
};


int minimumDist(int dist[], bool Tset[]) 
{
	int min=INT_MAX,index;
              
	for(int i=0;i<6;i++) 
	{
		if(Tset[i]==false && dist[i]<=min)      
		{
			min=dist[i];
			index=i;
		}
	}
	return index;
}

void Dijkstra(int graph[6][6],int src) 
{
	int dist[6];                         
	bool Tset[6];
	for(int i = 0; i<6; i++)
	{
		dist[i] = INT_MAX;
		Tset[i] = false;	
	}
	
	dist[src] = 0;               
	
	for(int i = 0; i<6; i++)                           
	{
		int m=minimumDist(dist,Tset);
		Tset[m]=true;
		for(int i = 0; i<6; i++)                  
		{
			if(!Tset[i] && graph[m][i] && dist[m]!=INT_MAX && dist[m]+graph[m][i]<dist[i])
				dist[i]=dist[m]+graph[m][i];
		}
	}
	cout<<"Vertex\t\tDistance from source"<<endl;
	for(int i = 0; i<6; i++)                      
	{ 
		char str=65+i; 
		cout<<str<<"\t\t\t"<<dist[i]<<endl;
	}
}

int main(){
	Graph graph(5);
	/*
	graph.insertEdgeDirected(0, 1);
	graph.insertEdgeDirected(1, 2);
	//graph.insertEdge(1, 2);
	//graph.insertEdge(2, 0);
	graph.insertEdgeDirected(2, 3);
	//graph.insertEdge(3, 3);*/
	graph.insertEdgeUndirected(1, 0);
	graph.insertEdgeUndirected(0, 2);
	graph.insertEdgeUndirected(2, 1);
	graph.insertEdgeUndirected(0, 3);
	graph.insertEdgeUndirected(3, 4);
	std::cout<<(graph.isCyclic(0))<<std::endl;
}