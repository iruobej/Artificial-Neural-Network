//Import statements
import com.opencsv.exceptions.CsvException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import com.opencsv.CSVReader;
import java.io.FileReader;
import java.util.List;
public class Main {

    public static int getPredictors() {
        return predictors;
    }

    private static int predictors = 8; //Because I am using 8 predictors in my data sets
    public static double[] inputValues = new double[predictors];
    public static double lparam = 0.1; //Because I am using 8 predictors in my data sets
    static double correctOutput; //Because I am using 8 predictors in my data sets
    static int epochs = 20000; // To hold the number of epochs
    public static void main(String[] args) {
        try {
            List<String[]> allData = readCsvData("C:\\Users\\joshu\\OneDrive\\Documents\\AI csw\\training.txt"); //To recieve as input the number of hidden nodes the user wants to have in the ANN
            int numberOfHiddenNodes = getNumberOfHiddenNodes();
            //Creating an instance of my backpropagation class
            Backpropagation backpropagation = new Backpropagation(numberOfHiddenNodes, allData, inputValues, correctOutput, lparam);
            backpropagation.initialiseWeightsandBiases();
            for (int i = 1; i < epochs+1; i++){
                // Annealing
                double startp = 0.1; // start learning parameter
                double endp = 0.01;// end learning parameter
                lparam = endp + (startp - endp) * (1 - (1 / (1 + Math.exp(10  - (double) (20 * i) / epochs))));
                for (String[] row : allData){
                    correctOutput = Double.parseDouble(row[predictors]);
                    //Forward pass
                    backpropagation.forwardPass(row);
                    backpropagation.backwardPass(correctOutput, row);
                }


                // To output MSE to the console every 100 epochs
                double totalOM = 0;
                int rowtotal = 0;
                if (i % 100 == 0 || i == 1 || i == epochs){   // &&
                    List<String[]> validData = readCsvData("C:\\Users\\joshu\\OneDrive\\Documents\\AI csw\\validation.txt");
                    //Finding the number of rows in my data set
                    for (String[] row : validData){
                        rowtotal += 1;
                    }
                    for (String[] row : validData){
//                backpropagation.forwardPass(inputs);
                        double[] inputs = new double[getPredictors()];
                        double actualOutput = Double.parseDouble(row[getPredictors()]);
                        double predictedOutput = backpropagation.forwardPass(row);
                        for (int j = 0; j < getPredictors(); j++) {
                            inputs[j] = Double.parseDouble(row[j]);
                        }
                        totalOM += Math.pow(actualOutput - predictedOutput, 2);
                        //MSE equation
                    } System.out.println(totalOM / rowtotal);
                    //System.out.println(lparam);
                }

            }

            //Reading my test set
            List<String[]> testData = readCsvData("C:\\Users\\joshu\\OneDrive\\Documents\\AI csw\\testing.txt");
            //System.out.println("-------------------These are the actual modelled output--------------");
            for (String[] row : testData){
                double[]inputs = new double[getPredictors()];
                for (int i = 0; i < getPredictors(); i++) {
                    inputs[i] = Double.parseDouble(row[i]);
                }

                double actualOutput = Double.parseDouble(row[getPredictors()]);
                double predictedOutput = backpropagation.forwardPass(row);
                //System.out.println(actualOutput);
            }

        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }
    }
    //Methods to get the number of hidden nodes, with error handling
    public static int getNumberOfHiddenNodes() {
        int numberOfHiddenNodes;
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the number of hidden nodes: ");


        while (true) {
            while (!scanner.hasNextInt()) {
                System.out.println("Please enter a valid integer.");
                scanner.next();
            }

            numberOfHiddenNodes = scanner.nextInt();
            if (numberOfHiddenNodes >= predictors / 2 && numberOfHiddenNodes <= predictors * 2) {
                break;
            } else {
                System.out.println("Please enter a number of hidden nodes between " + predictors / 2 + " and " + (predictors * 2) + ".");
            }
        }
        scanner.close();
        return numberOfHiddenNodes;
    }
    //Method to read data from a given file
    public static List<String[]> readCsvData(String filePath) throws IOException, CsvException {
        FileReader fileReader = new FileReader(filePath);
        CSVReader csvReader = new CSVReader(fileReader);
        List<String[]> allDataList = csvReader.readAll();
        List<String[]> a=new ArrayList<>();
        for (String[] row : allDataList){
            for (String value:row)
            {
                a.add(value.split("\\|"));
            }
        }
        csvReader.close();
        fileReader.close();
        return a;
    }
}
