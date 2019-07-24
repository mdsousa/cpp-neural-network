# cpp-neural-network
C++11, C++14, C++17, Deep Neural Net, DNN, Machine Learning

I've utilized C++ techniques presented by [Jason Turner's C++ Weekly](https://www.youtube.com/playlist?list=PLs3KjaCtOwSZ2tbuV1hx8Xz-rFZTan2J1) series.

Fully connected, simple neural network in modern c++

To compile on Compiler Explorer:
To compile for either the ARM Cortex-a72 (new raspberry pi) use gcc 8.2 (with -Os -std=c++17 -mtune=cortex-a72 or use -O3, which may give faster code, but more assembly) or for the X86-64, use gcc 9.1 (with -Os -std=c++17/2a -Wall -Wextra).

Need to complete feedForward function, followed by backPropagation function, along with utilities.
