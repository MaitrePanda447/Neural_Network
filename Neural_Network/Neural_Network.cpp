#include <iostream>
#include <vector>
#include <cmath>
#include <random>

double sigmoid(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivation(double activation)
{
	return activation * (1.0 - activation);
}

double nodeCost(double output, double expected)
{
	double error = expected - output;
	return error * error;
}

double totalCost(const std::vector<double>& outputs, const std::vector<double>& expected)
{
	double cost = 0.0;
	for (size_t i = 0; i < outputs.size(); ++i)
	{
		cost += nodeCost(outputs[i], expected[i]);
	}
	return cost;
}

double nodeCostDerivation(double output, double expected)
{
	return 2.0 * (output - expected);
}

struct Layer
{
	int numInputs;
	int numOutputs;

	std::vector<double> activations;
	std::vector<double> biais;
	std::vector<std::vector<double>> weights;

	std::vector<double> nodeValues;

	Layer(int inCount, int outCount)
		: numInputs(inCount), numOutputs(outCount),
		activations(outCount, 0.0),
		biais(outCount, 0.0),
		weights(outCount, std::vector<double>(inCount)),
		nodeValues(outCount, 0.0)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dist(-1.0, 1.0);

		for (int out = 0; out < numOutputs; ++out)
		{
			biais[out] = dist(gen);
			for (int in = 0; in < numInputs; ++in)
			{
				weights[out][in] = dist(gen);
			}
		}
	}

	std::vector<double> calculateOutputs(const std::vector<double>& inputs)
	{
		for (int out = 0; out < numOutputs; ++out)
		{
			double weightedInput = biais[out];
			for (int in = 0; in < numInputs; ++in)
			{
				weightedInput += inputs[in] * weights[out][in];
			}
			activations[out] = sigmoid(weightedInput);
		}
		return activations;
	}

	void calculateOutputLayerNodeValues(const std::vector<double>& expectedOutputs) {
		for (int i = 0; i < numOutputs; ++i) {
			double costDeriv = nodeCostDerivation(activations[i], expectedOutputs[i]);
			double activationDeriv = sigmoidDerivation(activations[i]);
			nodeValues[i] = costDeriv * activationDeriv;
		}
	}

	void calculateHiddenLayerNodeValues(const Layer& nextLayer) {
		for (int i = 0; i < numOutputs; ++i) {
			double newNodeValue = 0.0;
			for (int nextNode = 0; nextNode < nextLayer.numOutputs; ++nextNode) {
				double weightedInputDeriv = nextLayer.weights[nextNode][i];
				newNodeValue += nextLayer.nodeValues[nextNode] * weightedInputDeriv;
			}
			nodeValues[i] = newNodeValue * sigmoidDerivation(activations[i]);
		}
	}

	void applyGradients(const std::vector<double>& inputs, double learningRate) {
		for (int out = 0; out < numOutputs; ++out) {
			for (int in = 0; in < numInputs; ++in) {
				double weightGradient = nodeValues[out] * inputs[in];
				weights[out][in] -= learningRate * weightGradient;
			}
			double biasGradient = nodeValues[out] * 1.0;
			biais[out] -= learningRate * biasGradient;
		}
	}
};

class NeuralNetwork
{
public:
	std::vector<Layer> layers;

	NeuralNetwork(const std::vector<int>& layerSizes)
	{
		for (size_t i = 0; i < layerSizes.size() - 1; ++i)
		{
			layers.emplace_back(layerSizes[i], layerSizes[i + 1]);
		}
	}

	std::vector<double> calculateOutputs(std::vector<double> inputs) {
		for (auto& layer : layers) {
			inputs = layer.calculateOutputs(inputs);
		}
		return inputs;
	}

	void training(const std::vector<double>& inputs, const std::vector<double>& expected, double learningRate) {
		calculateOutputs(inputs);

		Layer& outputLayer = layers.back();
		outputLayer.calculateOutputLayerNodeValues(expected);

		for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
			layers[i].calculateHiddenLayerNodeValues(layers[i + 1]);
		}

		for (size_t i = 0; i < layers.size(); ++i) {
			const std::vector<double>& layerInputs = (i == 0) ? inputs : layers[i - 1].activations;
			layers[i].applyGradients(layerInputs, learningRate);
		}
	}
};

int main() {
	NeuralNetwork nn({ 2, 4, 1 });

	std::vector<std::vector<double>> inputs = { { 0, 0 },{ 0, 1 },{ 1, 0 },{ 1, 1 } };
	std::vector<std::vector<double>> expected = { { 0 },{ 1 },{ 1 },{ 0 } };
	double learningRate = 0.5;

	std::cout << "--- AVANT ENTRAINEMENT ---\n";
	for (size_t i = 0; i < inputs.size(); ++i) {
		auto out = nn.calculateOutputs(inputs[i]);
		std::cout << inputs[i][0] << " XOR " << inputs[i][1]
			<< " => " << out[0] << " (Attendu: " << expected[i][0] << ")\n";
	}

	std::cout << "\n--- ENTRAINEMENT ---\n";
	for (int epoch = 0; epoch <= 10000; ++epoch) {
		double epochCost = 0.0;
		for (size_t i = 0; i < inputs.size(); ++i) {
			nn.training(inputs[i], expected[i], learningRate);
			epochCost += totalCost(nn.calculateOutputs(inputs[i]), expected[i]);
		}
		if (epoch % 100 == 0) {
			std::cout << "Epoch " << epoch << " | Cout total : " << epochCost << "\n";
		}
	}

	std::cout << "\n--- APRES ENTRAINEMENT ---\n";
	for (size_t i = 0; i < inputs.size(); ++i) {
		auto out = nn.calculateOutputs(inputs[i]);
		std::cout << inputs[i][0] << " XOR " << inputs[i][1]
			<< " => " << out[0] << " (Attendu: " << expected[i][0] << ")\n";
	}

	return 0;
}