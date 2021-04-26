#include <chrono>
#include <iostream>
#include <bits/stdc++.h>
#include <omp.h>
#include <atomic>
#include <pthread.h>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <tuple>
#include <vector>
#include <thread>
#include <sys/mman.h>
#include <f77blas.h>
#include <cblas.h>

using namespace std;

//std::atomic<float> variable(0.0f);

void matmul(float* A, float* B, float* C, int m, int n, int p){
    #pragma omp parallel for deafult(none) shared(A, B, C, m, n, p)
        for(int i=0; i<m; i++){
            for(int j=0; j<p; j++){
                float result = 0.0f;
                #pragma omp simd
                    for(int k=0; k<n; k++){
                        result += *(A + i*m + k) * *(B + k*p + j);
                    }
                *(C + i*m + j) = result;
            }
        }
}

template <typename type>
class List{
public:
    List(){
        this->num_elements = 0;
        this->initialized = 0;
    }

    List(type input){
        this->array = (type*)malloc(sizeof(type));
        *(array) = input;
        this->initialized = 1;
        this->num_elements = 1;
    }

    void append(type item){
        if(this->initialized){
        size_t size = getSize();
        this->array = (type*)realloc(this->array, size + 1*sizeof(type));

        if(this->array == nullptr){
            cout<<endl<<"Failed to realloc, freeing the array pointer"<<endl;
            free(this->array);
        }

        *(this->array + size) = item;
        this->num_elements += 1;

        }

        else{
            this->array = (type*)malloc(sizeof(type));
            *(array) = item;
            this->initialized = 1;
            this->num_elements += 1;
        }

    }

    size_t getSize(){
        auto size = this->num_elements;
        return size;
    }

    type* data(int offset = 0){
        assert(offset < this->getSize() - 1);

        type* pointer = this->array + offset;
        return pointer;
    }

    /*
    :params = value to be removed, a function you want to use for comparing two values, defaults to nullptrs, if it is nullptr, "==" will be used.
    The custom function should be return an int (1 or 0 for true and false respectively) and should take type value, type value as params to compare. 
    It just removes the first value it encounters. remove_all parameter removes all occurences if 1 otherwise removes the first occurence. 
    Returns 1 if successful otherwise 0. I have to use mremap instead of realloc. Maybe it the memory not in use might be freed. (Have to check once.)
    */

    int remove(type item, int (*compare_func)(type v1, type v2) = nullptr, int remove_all=0){
        auto size = (int)this->getSize();
        int i;
        std::queue<int> occurences;
        if(compare_func == nullptr){
            #pragma omp simd
            for (i = 0; i<size; i++){
                if(*(this->array + i) == item)
                    occurences.push(i);
            }
            if (remove_all){
                int j = 0;
                while(!occurences.empty()){
                    i = occurences.front();
                    std::copy(this->array + i + 1 -j , this->array + size - j, this->array + i -j);
                    this->num_elements -=1 ;
                    occurences.pop();
                    this->array = (type*)realloc(this->array, this->num_elements*(sizeof(type)));
                    //free(this->array + size - j -1);
                    j+=1;
                }
                
            }
            else{
                i = occurences.front();
                std::copy(this->array + i + 1, this->array + size, this->array + i);
                this->array = (type*)realloc(this->array, (size-1)*sizeof(type));
                this->num_elements -= 1;
                //free(this->array + this->num_elements - 1);
            }
            return 0;
        }

        else{
            #pragma omp simd
            for(i = 0; i<size; i++){
                if(compare_func(*(this->array + i), item))
                    occurences.push(i);
            }
            if (remove_all){
                int j = 0;
                while(!occurences.empty()){
                    i = occurences.front();
                    std::copy(this->array + i + 1 -j, this->array + size - j, this->array + i -j);
                    occurences.pop();
                    this->num_elements -=1;
                    this->array = (type*)realloc(this->array, this->num_elements*(sizeof(type)));
                    //free(this->array + size - j - 1);
                    j+=1;
                }
            }
            else{
                i = occurences.front();
                std::copy(this->array + i + 1, this->array + size, this->array + i);
                this->array = (type*)realloc(this->array, (size-1)*sizeof(type));
                this->num_elements -= 1;
                //free(this->array + this->num_elements - 1);
            }
            return 0;
        }

        return 1;
    }

private:
    type* array;
    int initialized;
    int num_elements;
};


void ReLU(float* input, int size){
    #pragma omp simd
    for(int i = 0; i<size; i++){
        float element = *(input + i);
        int sign = (int32_t(element)>>31) + 1;
        *(input + i) = sign * element;
    }
}

void leaky_Relu(float* input, int size){
    #pragma omp simd
    for(int i = 0; i<size; i++){
        float element = *(input+i);
        float scaling_factor = 1.0f/2.0f;
        auto sign =  float((int32_t(element)>>31))+2.0f;
        //cout<<sign<<endl;
        *(input + i) = sign * element * scaling_factor;
    }
}

class Linear{
public:
    Linear(int input_size, int output_size){

        std::random_device device{};
        std::normal_distribution<float> distribution{0, 2};

        std::mt19937 generator{device()};
        this->input_neurons = input_size;
        this->output_neurons = output_size;

        this->weights = (float*)malloc(this->input_neurons * this->output_neurons * sizeof(float));
        this->bias = (float*)malloc(this->output_neurons * sizeof(float));
#pragma omp parallel for
        for(int i = 0; i < this->input_neurons; i++){
        #pragma omp simd
            for(int j = 0; j<this->output_neurons; j++){
                *(this->weights + i*this->input_neurons + j) = distribution(generator);
            }
        }

#pragma omp parallel for
        for(int i = 0; i<this->output_neurons; i++){
            *(this->bias + i) = distribution(generator); 
        }
    }


    float* forward(float* input, int batch_size){
        this->current_input = input;
        float* output = (float*)malloc(batch_size * this->output_neurons * sizeof(float));

            for(int i = 0; i< batch_size; i++){
                float* per_batch_input = input + this->input_neurons*i;

                #pragma omp parallel for
                {
                    for(int j=0; j<output_neurons; j++){
                        float result = 0.0f;
                        #pragma omp simd
                        for(int k=0; k< this->input_neurons; k++){
                            result += *(per_batch_input + k) * *(this->weights + k*this->output_neurons + j);
                        }
                        *(output + i*this->output_neurons + j) = result + *(this->bias+j);
                    }
                }
            }
            return output;
    }


    void backward(float* dldY, int batch_size){
        /*
            dldY is the upstream gradient calculated by dL/dY where Y is the scalar and Y the output matrix
            of shape batch x output_shape. dL/dW and dL/dX can be calculated as dL/dY*(W_Transpose) and 
            X^T*{}
        */
       this->dldX = (float*)malloc(batch_size * this->input_neurons * sizeof(float));
       this->dldW = (float*)malloc(this->input_neurons * this->output_neurons * sizeof(float));
       float* w_t = (float*)malloc(this->input_neurons * this->output_neurons * sizeof(float));
       float* x_t = (float*)malloc(batch_size * this->input_neurons * sizeof(float));

    #pragma omp parallel for
    for(int i =0; i<batch_size; i++){
        #pragma omp simd
            for(int j=0; j<this->input_neurons; j++){
                *(x_t + j*batch_size + i) = *(this->current_input + i*batch_size + j);
            }
    }
    #pragma omp parallel for
       for (int i = 0; i<this->input_neurons; i++){
        #pragma omp simd
           for (int j=0; j<this->output_neurons; j++){
               *(w_t + j*input_neurons + i) = *(this->weights + i*this->output_neurons + j);
           }
       }
       // Calculating dL/dW as dL/dY * W^T
        for(int i =0; i<batch_size; i++){
            float* per_batch_gradient = dldY + i*this->output_neurons;

        #pragma omp parallel for
            for(int j=0; j<this->input_neurons; j++){
                float result = 0.0f;
            #pragma omp simd
                for (int k=0; k<this->output_neurons; k++){
                    result += *(per_batch_gradient + k) * *(w_t + k*this->input_neurons + j);
                }
                *(this->dldW + i*this->input_neurons + j) = result;
            }
        }

        // Calculating dL/dX as x^T * dL/dY; [input x batch] * [batch * output]
    #pragma omp parallel for
        for(int i=0; i<this->input_neurons; i++){
            for(int j=0; j<this->output_neurons; j++){
                float result = 0.0f;
                for(int k=0; k<batch_size; k++){
                    result +=  *(x_t + i*batch_size + k) * *(dldY + k*this->output_neurons +j);
                }
                *(this->dldX + i*this->input_neurons + j) = result;
            }
        }
        this->grad[0] = this->dldW;
        this->grad[1] = this->dldX;
    }



private:
    int input_neurons;
    int output_neurons;
    float* weights;
    float* bias;
    float* current_input;
    float* grad[2]; // dL/dX and dL/dW 
    float* dldX, *dldW; // batch * input_neurons || input_neurons * output_neurons; 
};


class LinearRegression{
public:
    LinearRegression(int input_size){
        std::random_device device{};
        std::normal_distribution<float> distribution{0, 2};

        std::mt19937 generator{device()};

        this->input_size = input_size;
        this->bias = 1.0f;
        this->weights = (float*)malloc(input_size * sizeof(float));
        for (int i = 0; i < input_size; ++i) {
            *(this->weights+i) = distribution(generator);
        }
    }

    float forward(float* _input){
        this->input = _input;
        float result = 0.0f;
        float intermediate = 0.0;
        int i = 0;
        auto model_weights = this->weights;
#pragma omp parallel default(none) private(intermediate) shared(result, model_weights, _input)
        {
#pragma omp for
            for (i = 0; i < this->input_size; ++i) {
                intermediate = intermediate + *(this->weights + i) * *(_input + i);
            }
#pragma omp atomic
        result += intermediate;
    }
        result = result + this->bias;
        return result;
}

void backward(float loss, float lr){
#pragma omp simd
    for (int i = 0; i < this->input_size; ++i) {
        *(this->weights + i)  = *(this->weights + i) - 2*lr*loss*(*(this->input + i));
    }
    this->bias = this->bias - 2*lr*loss;
    }

private:
    float* weights;
    float  bias;
    int input_size;
    float* input;
};



class Convolution2D{
public:
    Convolution2D(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    {
        std::random_device device{};
        std::normal_distribution<float> distribution{0, 2};
        std::mt19937 generator{device()};
        
        this->in_channels = in_channels;
        this->out_channels = out_channels;
        this->padding = padding;
        this->kernel_size = kernel_size;
        this->stride = stride;

        for(int j=0; j<this->out_channels; j++){
            this->Kernels.push_back((float*)malloc(this->in_channels * kernel_size * kernel_size * sizeof(float)));
        }

    #pragma omp parallel for
        for(int i =0; i<in_channels; i++){
            float* kernel_plane = this->Kernels.at(i) + i*kernel_size*kernel_size;

            for(int j=0; j<kernel_size; j++){
            #pragma omp simd
                for(int k=0; k<kernel_size; k++){
                    *(kernel_plane + j*kernel_size + k) = distribution(generator);
                }
            }
        }

        this->bias = (float*)malloc(this->out_channels * sizeof(float));

#pragma omp parallel for
    for(int j=0; j<this->out_channels; j++){
        *(this->bias + j) = distribution(generator);
    }

}
    float* forward(float* input, int batch, int width, int height){

        omp_set_num_threads(12);
        
        int dx = this->kernel_size/2;
        int dy = this->kernel_size/2; 

        this->current_input = input;
        float* output = (float*)calloc(batch * this->out_channels * width * height, sizeof(float));

        for(int i=0; i<batch; i++){
            auto image_per_batch = input + i*this->in_channels*width*height;
  
            for (int j = 0; j<this->out_channels; j++){
                float* kernel = this->Kernels.at(j);

                for(int k = 0; k<this->in_channels; k++){
                    float* kernel_slice = kernel + k* this->kernel_size * this->kernel_size;
                    float* image_slice = image_per_batch + k*width*height;
   
                    for(int w = this->kernel_size; w < width - this->kernel_size; w++){
                        for(int h = this->kernel_size; h < height - this->kernel_size; h++){
                            float pixel_value = 0.0f;

                            for(int l = 0; l<this->kernel_size; l++){
                                for(int m = 0; m<this->kernel_size; m++){
                                    int x = h - dx + l;
                                    int y = i - dy + k;
                                    pixel_value += *(input + x*width + y) * *(kernel + l*this->kernel_size + m);
                                }
                            }
                            *(output + i*this->out_channels * width * height + j*width*height + w*width + h) += pixel_value;
                        }
                    }
                }
            }
        }

        return output;
    }
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    float* bias;
    float* current_input;
    std::vector<float*> Kernels;
};


class exampleClass{
public: 
    exampleClass(int n){
        cout<<"Created Class"<<endl;
        this->number = n;
    }

    ~exampleClass(){
        cout<<endl<<"This class is destroyed Now"<<endl;
    }

    template<typename t> t 
    addNum(t a, t b){
        return a + b;
    }

    void print(){
        cout<<this->number<<endl;
    }
private:
    int number;
};

   int variable = 1020;

void addNumber(int number){
    variable += number;
}

class templateClass{
public:
    templateClass(){
        cout<<"created"<<endl;
    }

    virtual ~templateClass(){
        cout<<"Destructed"<<endl;
    }

    virtual int add(int a, int b) final{
        return a+b;
    }
};

class derivedClass : public templateClass{
public:
    derivedClass(){
        cout<<"Contructed derivedClass"<<endl;
    }

    ~derivedClass(){
        cout<<"Destructed derivedClass";
    }

    int add(float a, float b){
        return a + b;
    }
};


class Base{
public:
    Base(){
        std::cout<<"Created Base Class"<<std::endl;
    }
    virtual ~Base(){}

    virtual void print() = 0;
    virtual int returnInt(int n) = 0;

};

class Derived : public Base {
public:
    Derived(){
        std::cout<<"Created Derived Class"<<std::endl;
    }
    void print(){
        std::cout<<"In print function"<<std::endl;
    }
    void print(int n){
        std::cout<<n<<std::endl;
    }
    int returnInt(int n){
        return n;
    }
    ~Derived(){
        std::cout<<"Destructing Derived Class now"<<std::endl;
    }
};


float max(float a, float b){
    std::cout<<"In float max"<<std::endl;
    return (a > b) ? a : b;
}

int max(int a, int b){
    std::cout<<"In Int max"<<std::endl;
    return (a > b) ? a : b;
}

template<typename type1, typename type2>
type1 compareNumber(type1 a, type1 b, type2 function){
    return function(a, b);
}



int main(){
    /*
    LinearRegression linearRegression(5000);
    auto input = (float*)malloc(5000 * sizeof(float));
    for (int i = 0; i < 5000; ++i) {
        *(input + i) = rand();
    }
    */
    //float output = linearRegression.forward(input);
    //cout<<endl<<output<<endl;
    //linearRegression.backward(20.0f, 1e-2);
    
    //ReLU(input, 5000);
    //std::atomic<float> a{};
    /*
    Linear linear(32, 128);
    input = (float*)malloc(3 * 32 * sizeof(float));

    #pragma omp parallel for
        for(int i =0; i<3; i++){
            #pragma omp simd
                for(int j =0; j<32; j++){
                    *(input + i*32 + j) = 1.0f;
            }
        }
    
    linear.forward(input, 3);
    */

   /*
   float a = 1.0f;
   float b = -98.034f;
   exampleClass* object = new exampleClass(60);

   std::thread t1(addNumber, 123);
   std::thread t2(addNumber, 205);
   addNumber(1244);
   t1.join();
   t2.join(); 
   cout<<variable<<endl;*/
   /*
   int v = 987;
   std::thread t([](int v){
       cout<<v<<endl;
   }, v);
   t.join(); */
   /*
   templateClass* ptr;
   derivedClass d;
   ptr = &d;
   cout<<ptr->add(1.0f, 2.0f);*/
   /*
   Convolution2D conv2D(32, 64, 5, 1, 1);
   float* input = (float*)malloc(5 * 32 * 400 * 400 * sizeof(float));

   for(int i = 0; i < 5; i++){
       for(int j=0; j<32; j++){
           #pragma omp parallel for
           for(int k=0; k<400; k++){

               #pragma omp simd
               for(int l=0; l<400; l++){
                   *(input + i*32*400*400 + j*400*400 + k*400 +l) = i + j + k + l;
               }
           }
       }
   }

   auto start_time = std::chrono::high_resolution_clock::now();
   float* output = conv2D.forward(input, 5, 400, 400);
   auto end_time = std::chrono::high_resolution_clock::now();

   auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
   cout<<duration.count()<<endl;

      for(int i = 0; i < 5; i++){
       for(int j=0; j<64; j++){
           #pragma omp parallel for
           for(int k=0; k<400; k++){

               #pragma omp simd
               for(int l=0; l<400; l++){
                   cout<<*(output + i*64*400*400 + j*400*400 + k*400 +l)<<"  ";
               }
           }

           cout<<endl;
           exit(10);
       }
   }

   cout<<endl;
    */
   /*
    auto a = (float*)malloc(100 * 500 * sizeof(float));
    auto b = (float*)malloc(500 * 750 * sizeof(float));
    auto c = (float*)calloc(100 * 750, sizeof(float));
    float count = 0.0f;

    for(int i=0; i<100; i++){
        for(int j=0; j<500; j++){
            *(a + i*100 + j) = 1.0f;
        }
    }

    for(int i=0; i<500; i++){
        for(int j=0; j<750; j++){
            *(b + i*500 + j) = 2.0f;
        }
    }

    matmul(a, b, c, 100, 500, 750);
    for(int i=0; i<50; i++){
        std::cout<<*(c+i)<<"  ";
    }
    */
    /*
    Derived derived;
    Derived * ptr = new Derived;
    //ptr->print();
    //ptr->print(10);
    //std::cout<<ptr->returnInt(100)<<std::endl;
    ptr->~Base();
    derived.~Base();*/
    
    //std::shared_ptr<Derived> ptr = std::make_shared<Derived>();
    //ptr->print();

    std::cout<<(compareNumber<float, float (float, float )>(5.2542f, 2.314f, max))<<std::endl;
    std::cout<<compareNumber<float>(1.3467f, 2.6721f, [=](float a, float b){
        return (a > b) ? a:b;
    })<<std::endl;

    std::cout<<max(5.3f, 2.7f)<<std::endl;
    std::cout<<max(1, 2)<<std::endl;
}
