#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

class TrainingData {
public:
	TrainingData(const string filename);
	bool isEof(void){
		return m_trainingDataFile.eof();
	}
	void getTopology(vector<unsigned> &topology);
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology){
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if(this->isEof() || label.compare("topology:") != 0){
		abort();
	}
	while(!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename){
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals){
	inputVals.clear();
	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);
	string label;
	ss >> label;
	if(label.compare("in:") == 0){
		double oneValue;
		while(ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}
	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals){
	targetOutputVals.clear();
	string line;
	getline(m_trainingDataFile, line);
		stringstream ss(line);
		string label;
		ss >> label;
		if(label.compare("out:") == 0){
			double oneValue;
			while(ss >> oneValue){
				targetOutputVals.push_back(oneValue);
			}
		}
		return targetOutputVals.size();
}

struct Connection{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val){ m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta;
	static double alpha;
	static string activation;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void){ return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
	
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;
// string Neuron::activation = "ReLu";
string Neuron::activation = "sigmoid";
// string Neuron::activation = "tanh";

void Neuron::updateInputWeights(Layer &prevLayer){
	// the weigts to be updated are in the connection container
	// in the neurons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n){
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		double newDeltaWeight = 
							// individual input, magnidied by the gradient and train rate;
							eta
							* neuron.getOutputVal()
							* m_gradient
							// step size of each update!
							+ alpha
							* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const{
	double sum = 0.0;

	// sum our contributions of the errors at the nodes we feed
	for(unsigned n = 0; n < nextLayer.size(); ++n){
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVals){
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){
	if(activation == "tanh"){
	//====  Hyperbolic Tangent =====//
	// tanh - output range [-1.0 1.0]
		x =  tanh(x);
	} else if(activation == "sigmoid") {
	//====  Sigmoid =====//
		// sigmoid - output range [0.0 1.0]
		x =  1/(1+exp(-x));
	} else if(activation == "ReLu"){
		x = max(0.0, x);
	}
	return x;
}

double Neuron::transferFunctionDerivative(double x){
	if(activation == "tanh"){
	//====  Hyperbolic Tangent =====//
	// tanh derivative
		x = 1.0 - tanh(x) * tanh(x);
	} else if(activation == "sigmoid"){
	//====  Sigmoid =====//
		x = (1/(1+exp(-x)))*(1 - (1/(1+exp(-x))));
	} else if(activation == "ReLu"){
		x = x > 0.0 ? 1.0 : 0.0;
	}
	return x;
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;

	// for the previous layer's outputs(which are our inputs)
	// include the bias node from the previous layer.
	for(unsigned n = 0; n < prevLayer.size(); ++n){
		// m_myIndex is the index to point all neurons in the previous layer.
		// we want to aggregate the values in previous layer's neurons
		sum += prevLayer[n].getOutputVal()*prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	// Accessing each neuron and store Connection data structure!
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

class Net {
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const{
	resultVals.clear();
	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n){
		double a = m_layers.back()[n].getOutputVal() > 0.5 ? 0.0 : 1.0;
		resultVals.push_back(a);
	}
}

void Net::backProp(const vector<double> &targetVals){
	// calculate overall net error [RMS of output neuron errors]
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
		// diff between target and prediction
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error = delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement(weighted average)
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
						 / (m_recentAverageSmoothingFactor + 1.0);
	
	// calculate output layer gradients
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// calculate gradients on hidden layer
	for(unsigned layerNum = m_layers.size() - 2; layerNum < m_layers.size(); --layerNum){
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum+1];
		for(unsigned n = 0; n < hiddenLayer.size(); ++n){
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}

	}

	// for all layers from outputs to first hidden layer
	// update connection weights
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n){
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals){
	// check if the size of input and first layer of the designed neural nets is matching
	assert(inputVals.size() == m_layers[0].size() - 1);
	// Assign (latch) the input values into the input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i){
		// defining the output of each neuron
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// forward propagation
	// skip input layer!
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		// for the sake of visibility to the layer, they can be propagated the preceeding layer's outputs
		// by defining prevLayer here!
		Layer &prevLayer = m_layers[layerNum - 1];

		// accessing each neuron except bias
		for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology){
	unsigned numLayers = topology.size();

	// layer num is the index number for layer
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){

		// append the layer
		m_layers.push_back(Layer());

		// check if the current layer in this loop has arrived at the output layer
		// if so, numOutputs becomes 0, otherwise, get the number of the neuron in the next layer
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		// we have made a new layer, now fill it "ith" neurons, and
		// add a bias neuron to the layer. so this way to loop through does this!
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			// back() gives us the most recent element in the container
			// add a bias to the layer after at the last neuorn!
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "made a neuron" << endl;
		}

		// force the bias node's output value to 1.0, it is the last neuron create above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

int main(int argc, char *argv[]){
	// topology defined the architecture of the net!!
	// e.g. (3,4,2) of DNN which is { 2 + bias, 4 + bias, 1 + bias } architecture of the net
	TrainingData trainData("../data/test.txt");
	vector<unsigned> topology;

	trainData.getTopology(topology);
	for(int i = 0; i < topology.size(); ++i){
		cout << topology[i] << endl;
	}

	Net myNet(topology);
	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while(!trainData.isEof()){
		++trainingPass;
		cout << endl << "Pass" << trainingPass;
		if(trainData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);
		myNet.getResults(resultVals);
		showVectorVals("Outputs: ", resultVals);
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());
		myNet.backProp(targetVals);
		cout << "Net recent average error: " << myNet.getRecentAverageError() << endl;
	}
	cout << endl << "Done!!" << endl;
}