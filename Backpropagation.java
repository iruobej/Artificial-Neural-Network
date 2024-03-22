//Import statements
import java.util.List;
import java.util.Random;

public class Backpropagation {
    //Upper and lower bound for weights and biases between input and hidden layer
    private final double Ilowerbound;
    double IupperBound;
    double lparam;
    //Upper and lower bound for weights and biases between hidden and output layer
    double lowerBound;
    double upperBound;
    int predictors = 8;
    int numberOfHiddenNodes;
    double correctOutput;
    double[] inputValues;
    //2d array for weights between the input the hidden layer
    double[][] weights;
    //array to store weights from the hidden layer to output node
    double[] wToOutput;
    double[] biases; // biases for hidden layer
    List<String[]> allData;
    //array to store the delta value for each of my hidden nodes
    private double[] hiddenDeltas;
    //To store the weighted sum of all input nodes for each hidden node
    private double[] weightedSums;
    //For holding the sigmoid values of each hidden node
    private double sigValues[];
    //For holding the sum of all the outputs of each hidden node, multiplied by their respective weight
    private double totalSumO;


    private double outDelta; //The delta for the output node
    Random random = new Random(); //Calling the Random class for initialising my weights and biases
    //Constructor

    public Backpropagation(int numberOfHiddenNodes, List<String[]> allData, double[] inputValues, double correctOutput, double lparam) {
        this.numberOfHiddenNodes = numberOfHiddenNodes;
        this.lowerBound = -2.0 / numberOfHiddenNodes;
        this.upperBound = 2.0 / numberOfHiddenNodes;
        this.Ilowerbound = -2.0 / predictors;
        this.IupperBound = 2.0 / predictors;
        this.weights = new double[predictors][numberOfHiddenNodes];
        this.wToOutput = new double[numberOfHiddenNodes];
        this.weightedSums = new double[numberOfHiddenNodes];
        this.biases = new double[numberOfHiddenNodes+1];//+1 for the output node bias
        this.hiddenDeltas = new double[numberOfHiddenNodes];
        this.sigValues = new double[numberOfHiddenNodes];
        this.inputValues = inputValues;
        this.outDelta = outDelta;
        this.allData = allData;
        this.correctOutput = correctOutput;
        this.totalSumO = 0;
        this.lparam = Main.lparam;
    }

    // Method to generate random value within the specified range of my upper and lower bound
    private double generateRandomValue() {
        return lowerBound + (upperBound - lowerBound) * random.nextDouble();
    }
    private double IgenerateRandomValue() {
        return Ilowerbound + (IupperBound - Ilowerbound) * random.nextDouble();
    }

    // Method to initialize weights with random values
    public void initialiseWeightsandBiases() {
        for (int i = 0; i < predictors; i++) {
            for (int j = 0; j < numberOfHiddenNodes; j++) {
                weights[i][j] = IgenerateRandomValue();
            }
        }
        for (int j = 0; j < numberOfHiddenNodes; j++) {
            biases[j] = IgenerateRandomValue();
            wToOutput[j] = generateRandomValue();
        }
        //For my bias for output node
        biases[numberOfHiddenNodes] = generateRandomValue();
    }
    // Each hidden node adding a bias to their respective weighted sums, and applying the sigmoid function
    public double forwardPass(String[] row){

        for (int i = 0; i < numberOfHiddenNodes; i++) {
            double weightedSum = 0;
            for (int j = 0; j < predictors; j++) {
                //add each input value multiplied by each weight to a grand total 'weightedSum'
                weightedSum += Double.parseDouble(row[j]) * weights[j][i];
            }
            weightedSums[i]=weightedSum;
            weightedSums[i] += biases[i];
            sigValues[i] = sigmoid(weightedSums[i]);
        }
        //For output node's output:
        totalSumO = 0;
        for (int i=0; i < numberOfHiddenNodes; i++){

            totalSumO += sigValues[i] * wToOutput[i];
        }
        totalSumO += biases[numberOfHiddenNodes];
        totalSumO = sigmoid(totalSumO);
        //System.out.println(totalSumO);
        return totalSumO;
    }
    //sigmoid function
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    //For calculating the differential for my delta equation
    public double differential(double x){
        return x * (1-x);
    }
    //Delta equation
    public double delta (double c, double x, double d){
        return (c - x) * d;
    }

    public void backwardPass (double correctOutput, String[] row){
        //calculate delta for the outputnode
        outDelta = delta(correctOutput, totalSumO, differential(totalSumO));
        //updating the bias for the output node
        biases[numberOfHiddenNodes] += (lparam * outDelta);
        //updating the hidden node deltas
        for (int i=0; i < numberOfHiddenNodes; i++){
            double differential = differential(sigValues[i]);

            hiddenDeltas[i] = wToOutput[i] * outDelta * differential;
        }

        //update weights from input to hidden
        for (int i=0; i < numberOfHiddenNodes; i++){
            for (int j=0; j < predictors; j++){
                //System.out.println(weights[i][j]);
                weights[j][i] = weights[j][i] + lparam * hiddenDeltas[i] * Double.parseDouble(row[j]);
            }
        }
        //updating hidden nodes' biases and weights
        for (int j=0; j < numberOfHiddenNodes; j++){
            wToOutput[j] += (lparam * outDelta * sigValues[j]);
            biases[j] = biases[j] + lparam * hiddenDeltas[j];
        }

    }
}
