#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

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

std::string trim(const std::string& str)
{
	size_t first = str.find_first_not_of(" \t\r\n");
	if (first == std::string::npos)
	{
		return "";
	}
	size_t last = str.find_last_not_of(" \t\r\n");
	return str.substr(first, last - first + 1);
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
		for (auto& layer : layers)
		{
			inputs = layer.calculateOutputs(inputs);
		}
		return inputs;
	}

	void training(const std::vector<double>& inputs, const std::vector<double>& expected, double learningRate) {
		calculateOutputs(inputs);

		Layer& outputLayer = layers.back();
		outputLayer.calculateOutputLayerNodeValues(expected);

		for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i)
		{
			layers[i].calculateHiddenLayerNodeValues(layers[i + 1]);
		}

		for (size_t i = 0; i < layers.size(); ++i)
		{
			const std::vector<double>& layerInputs = (i == 0) ? inputs : layers[i - 1].activations;
			layers[i].applyGradients(layerInputs, learningRate);
		}
	}
};

int main() {
	// --- ETAPE 1 : LECTURE ET NETTOYAGE DU FICHIER TXT ---
	std::map<std::string, int> itemToIndex;
	std::vector<double> prices;

	std::ifstream file("../liste_items_et_prix.txt");
	if (!file.is_open())
	{
		std::cerr << "Erreur : impossible d'ouvrir le fichier.\n";
		return 1;
	}

	std::string line;
	while (std::getline(file, line))
	{
		if (line.find(';') == std::string::npos)
		{
			continue;
		}

		std::stringstream ss(line);
		std::string nom, categorie, niveauStr, prixStr;

		if (std::getline(ss, nom, ';') &&
			std::getline(ss, categorie, ';') &&
			std::getline(ss, niveauStr, ';') &&
			std::getline(ss, prixStr, ';'))
		{
			nom = trim(nom);
			prixStr = trim(prixStr);

			try {
				double prix = std::stod(prixStr);
				itemToIndex[nom] = static_cast<int>(prices.size());
				prices.push_back(prix);
			}
			catch (...) {
				continue;
			}
		}
	}
	file.close();

	int N = static_cast<int>(prices.size());
	std::cout << N << " items charges depuis le fichier.\n";

	// --- ETAPE 2 : NORMALISATION DES PRIX (LOG10) ---
	double prixMax = 0.0;
	for (double p : prices)
	{
		if (p > prixMax)
		{
			prixMax = p;
		}
	}
	double maxLog = std::log10(prixMax) + 0.5;

	std::vector<std::vector<double>> expected(N, std::vector<double>(1));
	for (int i = 0; i < N; ++i)
	{
		expected[i][0] = std::log10(prices[i]) / maxLog;
	}

	// --- ETAPE 3 : CREATION DES VECTEURS ONE-HOT ET DU RESEAU ---
	std::vector<std::vector<double>> inputs(N, std::vector<double>(N, 0.0));
	for (int i = 0; i < N; ++i)
	{
		inputs[i][i] = 1.0;
	}

	NeuralNetwork nn({ N, 16, 1 });
	double learningRate = 0.4;
	int nbEpoques = 3000;

	// --- ETAPE 4 : ENTRAINEMENT ---
	std::cout << "\n--- ENTRAINEMENT ---\n";
	for (int epoch = 0; epoch <= nbEpoques; ++epoch)
	{
		for (int i = 0; i < N; ++i)
		{
			nn.training(inputs[i], expected[i], learningRate);
		}

		if (epoch % 500 == 0)
		{
			double epochCost = 0.0;
			for (int i = 0; i < N; ++i)
			{
				epochCost += totalCost(nn.calculateOutputs(inputs[i]), expected[i]);
			}
			std::cout << "Epoch " << epoch << " | Cout total : " << epochCost << "\n";
		}
	}

	// --- ETAPE 5 : INTERROGATION UTILISATEUR ---
	std::cout << "\n--- ESTIMATION DES PRIX ---\n";
	std::string saisie;

	while (true)
	{
		std::cout << "\nEntrez le nom exact d'un item (ou 'quit' pour quitter) : ";
		std::getline(std::cin, saisie);
		saisie = trim(saisie);

		if (saisie == "quit")
		{
			break;
		}

		if (itemToIndex.count(saisie) > 0)
		{
			int idx = itemToIndex[saisie];
			auto out = nn.calculateOutputs(inputs[idx]);

			long long prixPredit = static_cast<long long>(std::round(std::pow(10.0, out[0] * maxLog)));
			long long prixReel = static_cast<long long>(prices[idx]);

			std::cout << "-> Prix predit par l'IA : " << prixPredit << " Kamas\n";
			std::cout << "-> Prix reel du fichier : " << prixReel << " Kamas\n";
		}
		else
		{
			std::cout << "Objet introuvable dans la base.\n";
		}
	}

	return 0;
}