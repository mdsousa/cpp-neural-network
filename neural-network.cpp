#include <numeric>
#include <cmath>
#include <array>
#include <algorithm>
#include <iterator>
#include <iostream>


/*
//
// Simple, Fully connected Neural Net
//
// Compiled with ARM gcc 8.2 (with -Os -std=c++17 -mtune=cortex-a72 or use -O3 which may give faster but more assembly) or X86-64 gcc 9.1 (with -Os -std=c++17/2a -Wall -Wextra)
//
// Implemented with what is known as Modern C++ (11, 14, 17, 20 hopefully soon)
*/
const bool DebugWeights = false;
const bool DebugNodeOutputs = false;


//
// Data type used in NN
//
typedef float dataType;

// constexpr std::array<uint32_t, 3> topology{2,3,1}; // layers/nodes : pre c++17
constexpr std::array topology{2u,3u,1u}; // layers/nodes : template argument deduction done in C++17 : ex: 2*3+3*1=9; bias weights are accounted for in feedForward

//
// The number of weights for this net minus the bias
//
constexpr std::uint32_t numWeights{[] { std::uint32_t layerTopology{0}; std::uint32_t previousTopology{0}; std::uint32_t totalTopology{0}; for(auto const& x: topology) {previousTopology = layerTopology; layerTopology = x; totalTopology += layerTopology*previousTopology;} return totalTopology; }() };

//
// Each node uses a bias
//
dataType bias{1.0};

//
// Input data and ground truth for practice
//
const uint32_t numInputVals{10000};
const std::array<std::array<dataType, 3>, numInputVals> inputData {};


constexpr uint64_t seed()
{
   uint64_t shifted = 0;

   for( const auto c: __TIME__ )
   {
      shifted <<= 8;
      shifted |= c;
   }
   return shifted;
}

struct PCG
{
   struct pcg32_random_t { std::uint64_t state=0; std::uint64_t inc=seed(); };
   pcg32_random_t rng;
   typedef std::uint32_t result_type;
   constexpr result_type operator()()
   {
      return pcg32_random_r();
   }
   constexpr static result_type min()
   {
      return std::numeric_limits<result_type>::min();
   }
   constexpr static result_type max()
   {
      return std::numeric_limits<result_type>::max();
   }
private:
   constexpr std::uint32_t pcg32_random_r()
   {
      std::uint64_t oldstate = rng.state;
      rng.state = oldstate * 6364136223846793005ULL + (rng.inc|1);
      std::uint32_t xorshifted = ((oldstate >> 18u)^oldstate) >> 27u;
      std::uint32_t rot = oldstate >> 59u;
      return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
   }
};

constexpr auto call_random(std::size_t i)
{
   PCG pcg;
   while(i > 0) { pcg(); --i; }
   return pcg();
}


//
// neuron weights laid out in array whose size is based on numWeights
//
template<typename T>
constexpr auto getRandomWeights()
{
   std::array<T, numWeights> weights{};
   uint32_t i{2}; // call_random weights starts at 0, we want to avoid that
   dataType sumWeights = 0.0;
   [&]() { for(auto& x: weights) {x = static_cast<dataType>(call_random(i++)); sumWeights += x;} }();
   [&]() { for(auto& x: weights) {x = x/sumWeights;} }(); // normalize the weights to sum to 1.0
   return weights;
}


//
// Arrays of value used throughout the feed forward and back propagation process
//
constexpr std::array weightsMatrix(getRandomWeights<dataType>());
std::array<dataType, numWeights> targetValues {}; // the groundtruth forbackPropagation
std::array<dataType, numWeights> nodeOutputValues {}; // running outputs of the nodes
std::array<dataType, numWeights> nodeGradients {};

//
// Network made up of layers, which are inturn, made up of neurons
//
template <typename T>
class NeuralNetwork
{
public:
//   constexpr NeuralNetwork() : weightsMatrix(getRandomWeights<T>()) {}
   void backPropagation()
   {
      dataType delta{0}; // diff between target value and calculated value
      // The following is for running error average if wanted
/*      dataType rmsError{0}; // RMS of delta
      uint32_t outputIndex{numWeights-topology[topology.size()-1]}; // where do the outputs begin in array
      uint32_t targetIndex = targetValues.size()-1; // where do the targets end in array
      rmsError = sqrt(std::inner_product(targetValues.begin(), targetValues.begin()+targetIndex, nodeOutputValues.begin()+outputIndex, 0.0f)/nodeOutputValues[outputIndex]);*/

      uint32_t outputIndex{numWeights-topology[topology.size()-1]}; // where do the outputs begin in array
      uint32_t targetIndex{numWeights-topology[topology.size()-1]}; // where do the targets end in array
      for(uint32_t v = numWeights-topology[topology.size()-1]; v < numWeights-1; ++v) // take care of outputs
      {
         nodeGradients[v] = (targetValues[v]-nodeOutputValues[v])*(nodeOutputValues[v]>0 ? 1.0f : 0.0f); // derivative of RELU
      }
      for(uint32_t v = 0; v < numWeights-topology[topology.size()-1]; ++v) // take care of inner gradients
      {

      }
   }

   void feedForward()
   {
      if( topology.size() > 1 ) // check if we have more than an input layer
      {
          uint32_t index = topology[1]; // for ex: 2,3,1: 2 is number of input nodes, so start with 3
          for( std::size_t layer = 1; layer < topology.size(); ++layer )
          {
              uint32_t nodes(0.0); // number of nodes in this layer
              uint32_t previousNodes{0}; // number of nodes in previous layer
              nodes = topology[layer]; // get the number of nodes in this layer, for ex: 3,2,1
              previousNodes = topology[layer-1]; // need previous layer's values for feedForward calcs
              for( uint32_t node = 0; node < nodes; ++node, ++index )
              {
                  for(uint32_t previousNode = 0; previousNode < previousNodes; ++previousNode )
                  {
                      // sum of all previous layer's node's output values multiplied by this node's weight
                      nodeOutputValues[index] += nodeOutputValues[previousNode]*weightsMatrix[index];
                  }
                  nodeOutputValues[index] += bias*weightsMatrix[index];
                  nodeOutputValues[index] = std::max(0.0f, nodeOutputValues[index]); // apply transfer function -- RELU function
                  ++index;
              }
          }
      }
   }

   auto printWeights()
   {
      dataType total(0);
      std::cout.precision(9);
      std::cout << std::fixed;
      uint32_t index(0);
      uint32_t nodes(0);
  
      for(std::size_t layer = 0; layer < topology.size(); ++layer)
      {
         nodes = topology[layer]; // number of nodes in this layer
         for(uint32_t node = 0; node < nodes; ++node)
         {
            total += weightsMatrix[index];
            std::cout << weightsMatrix[index++];
            if(node < nodes-1) std::cout << ",";
         }
         std::cout << "\n";
      }
      std::cout << "total: " << total << "\n";
   }

   auto printOutputValues()
   {
      std::cout.precision(9);
      std::cout << std::fixed;
      uint32_t index(0);
      uint32_t nodes(0);
      for(std::size_t layer = 0; layer < topology.size(); ++layer)
      {
         nodes = topology[layer]; // number of nodes in this layer
         for(uint32_t node = 0; node < nodes; ++node)
         {
            std::cout << nodeOutputValues[index++];
            if(node < nodes-1) std::cout << ",";
         }
         std::cout << "\n";
       }
    }

private:
};

//
// Test the NN
//
int main()
{
   NeuralNetwork<float> nw{};

    std::cout << "numWeights: " << numWeights << "\n";
   if constexpr(DebugWeights)
   {
      nw.printWeights();
   }
   for(uint32_t outer = 0; outer < 10; ++outer)
   {
      for(uint32_t inner = 0; inner < topology[0]; ++inner)
      {
         nodeOutputValues[inner] = inputData[outer][inner];
         std::cout << "nodeOutputValues[" << inner << "], " << nodeOutputValues[inner] << "\n";
      }
      uint32_t outputIndex = numWeights - topology.size() - 1; // where the next output value is in the input array
      uint32_t numOutputValues = topology[topology.size() - 1]; // how many output values are expected for each feedforward
      // std::cout << "numOutputValues: " << numOutputValues << "\n";
      for(uint32_t targetIndex = 0; targetIndex < numOutputValues; ++targetIndex)
      {
         targetValues[targetIndex] = inputData[outer][outputIndex];
         std::cout << "inputData[" << outer << "][" << outputIndex << "] " << inputData[outer][outputIndex] << "\n";
         std::cout << "targetValues[" << targetIndex << "] " << targetValues[targetIndex] << " : ";
      }
      nw.feedForward();
      nw.backPropagation();
      if( outer < 10 || outer > 9990 )
      {
         std::cout.precision(9);
         // std::cout << std::fixed;
         std::cout << nodeOutputValues[0] << " " << nodeOutputValues[1] << ", " << nodeOutputValues[numWeights-1] << "\n";
      }
   }
   if constexpr(DebugNodeOutputs)
   {
      nw.printOutputValues();
   }
   return 0;
}

