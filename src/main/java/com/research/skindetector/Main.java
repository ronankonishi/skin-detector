package com.research.skindetector;

import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * The heart of the program that calls everything that needs to run.
 *
 * @author Ronan Konishi
 * @version 1.0
 */
public class Main {

    static int iterations = 1;
    static double learningRate = 0.001;
    static double momentum = 0.9;
    static double weightDecay = 0.005;

    static int rngseed = 123;
    static Random ranNumGen;
    static JsonImageRecordReader recordReader;

    static int height = 360;
    static int width = 360;
    static int nChannels = 3; // Number of input channels
    static int outputNum = 2; // The number of possible outcomes
    static int batchSize = 20;

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {
        File mixedData = new File("C:\\Users\\ronan\\Desktop\\test\\mixedData\\");
        File trainData = new File("C:\\Users\\ronan\\Desktop\\test\\trainData\\");
        File testData = new File("C:\\Users\\ronan\\Desktop\\test\\testData\\");
        NeuralNetwork network = new NeuralNetwork(mixedData, trainData, testData, rngseed, height, width, nChannels, batchSize, outputNum);

        network.buildNet(iterations, learningRate, momentum, weightDecay);

//        log.info("*****TRAIN MODEL********");
        network.train();
        network.UIenable();
//
//        log.info("*****SAVE TRAINED MODEL******");
//        network.saveBuild("trained_model.zip");
//
//        log.info("*****EVALUATE MODEL*******");
//        log.info(network.evaluate().stats());
    }
}