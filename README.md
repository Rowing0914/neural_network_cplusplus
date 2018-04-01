# Introduction
This is a simple implementation of NN in C++.  
Inspired by this vimeo tutorial [David Miller](https://vimeo.com/19569529).  

# Demo

$ g++ Make_Training_Sample.cpp -o Make_Training_Sample  
$ ./Make_Training_Sample > ../data/test.txt  
$ g++ Neural_Net.cpp -o Neural_Net  
$ ./Neural_Net  


# NN Architecture in this code
Input layer: 2 neurons  
Hidden layer: 4 neurons  
Output layer: 1 neuron

# Short Summary
1. Activate Function: sigmoid or tahn
2. Number of neurons are defined in Make_Training_Sample.cpp by "Topolpgy"

Keep updating.