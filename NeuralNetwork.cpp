#include "NeuralNetwork.hpp"
#include "Trace.hpp"
#include <algorithm>
using namespace std;

// NeuralNetwork -----------------------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::eval() {
    evaluating = true;
}

void NeuralNetwork::train() {
    evaluating = false;
}

void NeuralNetwork::setLearningRate(double lr) {
    learningRate = lr;
}

void NeuralNetwork::setInputNodeIds(std::vector<int> inputNodeIds) {
    NeuralNetwork::inputNodeIds = inputNodeIds;
}

void NeuralNetwork::setOutputNodeIds(std::vector<int> outputNodeIds) {
    NeuralNetwork::outputNodeIds = outputNodeIds;
}

vector<int> NeuralNetwork::getInputNodeIds() const {
    return vector<int>(inputNodeIds);
}

vector<int> NeuralNetwork::getOutputNodeIds() const {
    return vector<int>(outputNodeIds);
}

vector<double> NeuralNetwork::predict(DataInstance instance) {
    flush();
    vector<double> input = instance.x;

    if (input.size() != inputNodeIds.size()) {
        cerr << "input size mismatch." << endl;
        cerr << "\tNeuralNet expected input size: " << inputNodeIds.size() << endl;
        cerr << "\tBut got: " << input.size() << endl;
        return vector<double>();
    }

    for (int i = 0; i < inputNodeIds.size(); i++) {
        int nodeId = inputNodeIds[i];
        nodes[nodeId]->preActivationValue = input[i];
        nodes[nodeId]->postActivationValue = input[i];
    }

    vector<int> inDegree(size, 0);
    for (int i = 0; i < size; i++) {
        for (auto& pair : adjacencyList[i]) {
            inDegree[pair.first]++;
        }
    }

    vector<int> processedPreds(size, 0);

    queue<int> q;
    vector<bool> visited(size, false);
    for (int i = 0; i < inputNodeIds.size(); i++) {
        q.push(inputNodeIds[i]);
        visited[inputNodeIds[i]] = true;
    }

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        bool isInput = find(inputNodeIds.begin(), inputNodeIds.end(), current) != inputNodeIds.end();
        if (!isInput) {
            visitPredictNode(current);
        }

        for (auto& pair : adjacencyList[current]) {
            int neighborId = pair.first;
            Connection& conn = pair.second;
            visitPredictNeighbor(conn);
            processedPreds[neighborId]++;
            if (processedPreds[neighborId] == inDegree[neighborId] && !visited[neighborId]) {
                visited[neighborId] = true;
                q.push(neighborId);
            }
        }
    }

    vector<double> output;
    for (int i = 0; i < outputNodeIds.size(); i++) {
        int dest = outputNodeIds.at(i);
        output.push_back(nodes.at(dest)->postActivationValue);
    }

    if (evaluating) {
        flush();
    } else {
        batchSize++;
        contribute(instance.y, output.at(0));
    }
    return output;
}

bool NeuralNetwork::contribute(double y, double p) {
    contributions.clear();
    for (auto node : inputNodeIds) {
        contribute(node, y, p);
    }
    return true;
}

double NeuralNetwork::contribute(int nodeId, const double& y, const double& p) {
    visitContributeStart(nodeId);
    double incomingContribution = 0;
    double outgoingContribution = 0;

    if (contributions.count(nodeId) > 0) {
        return contributions[nodeId];
    }

    if (adjacencyList.at(nodeId).empty()) {
        outgoingContribution = -1 * ((y - p) / (p * (1 - p)));
    } else {
        for (auto& pair : adjacencyList[nodeId]) {
            Connection& conn = pair.second;
            incomingContribution = contribute(pair.first, y, p);
            visitContributeNeighbor(conn, incomingContribution, outgoingContribution);
        }
        bool isInput = find(inputNodeIds.begin(), inputNodeIds.end(), nodeId) != inputNodeIds.end();
        if (!isInput) {
            visitContributeNode(nodeId, outgoingContribution);
        }
    }

    contributions[nodeId] = outgoingContribution;
    return outgoingContribution;
}

bool NeuralNetwork::update() {
    for (int i = 0; i < size; i++) {
        NodeInfo* node = nodes[i];
        node->bias = node->bias - (learningRate * node->delta);
        node->delta = 0;
        for (auto& pair : adjacencyList[i]) {
            Connection& conn = pair.second;
            conn.weight = conn.weight - (learningRate * conn.delta);
            conn.delta = 0;
        }
    }
    flush();
    return true;
}






