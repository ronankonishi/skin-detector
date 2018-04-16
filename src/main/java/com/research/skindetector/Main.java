package com.research.skindetector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * The heart of the program that calls everything needed to run.
 *
 * @author Ronan Konishi
 * @version 1.0
 */
public class Main {

    static Random ranNumGen;
    static JsonImageRecordReader recordReader;


    //hyper parameters
    static double learningRate = 0.001;
    static double momentum = 0.9;
    static double weightDecay = 0.005;

    static int rngseed = 123;

    static int height = 360; //of image
    static int width = 360; //of image
    static int nChannels = 3; // Number of input channels
    static int outputNum = 2; // The number of possible outcomes
    static int batchSize = 20; //batch size for Stochastic Gradient Descent

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {
//        String trainedpath = "C:\\Users\\ronan\\Documents\\AP Research\\skin-detector\\trained_model.zip"; //uncomment if using an already trained model
        File mixedData = new File("C:\\Users\\ronan\\Desktop\\ISIC-images\\mixedData\\");
        File trainData = new File("C:\\Users\\ronan\\Desktop\\ISIC-images\\trainData\\");
        File testData = new File("C:\\Users\\ronan\\Desktop\\ISIC-images\\testData\\");
        NeuralNetwork network = new NeuralNetwork(mixedData, trainData, testData, rngseed, height, width, nChannels, batchSize, outputNum);

        network.buildNet(learningRate, momentum, weightDecay);

        log.info("*****TRAIN MODEL********");
        network.train();

        log.info("*****ENABLE UI********");
        network.UIenable();

//        log.info("*****SAVE TRAINED MODEL******");
//        network.saveBuild("trained_model.zip");

        log.info("*****EVALUATE MODEL*******");
        log.info(network.evaluate().stats());
    }
}