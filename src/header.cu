#include "../include/header.hpp"

namespace{

__global__ void kernel(){
}
}

void test(){
kernel<<<1,1>>>();
}
