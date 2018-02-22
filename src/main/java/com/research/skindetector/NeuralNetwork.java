package com.research.skindetector;

import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Convolutional Neural Network class that applies Supervised Learning.
 *
 * This is the main utility that builds, trains, and evaluates the neural network.
 * 
 * Based on deeplearning4j open source library and tutorials.
 *
 * @author Ronan Konishi
 * @version 1.0
 *
 */
public class NeuralNetwork {
    private static Logger log = LoggerFactory.getLogger(Main.class);
    File trainData, testData;
    int rngseed, height, width, channels, batchSize, outputNum;
    String netPath;
    Random ranNumGen;
    MultiLayerNetwork model;
    JsonImageRecordReader recordReader;
    DataNormalization scaler;
    DataSetIterator iter;


    /**
     * Constructor for non distinguished training and testing data.
     * Also for if need to create a neural network.
     *
     * @param mixedData Path to file with mixed data
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize
     * @param outputNum The number of nodes in the output layer
     */
    public NeuralNetwork(File mixedData, int rngseed, int height, int width, int channels, int batchSize, int outputNum) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        init();
        log.info("Building Neural Network from scratch...");
        buildNet();
    }

    /**
     * Constructor for non distinguished training and testing data.
     * Also for if importing an already built neural network.
     *
     * @param mixedData Path to file with mixed data
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize
     * @param outputNum The number of nodes in the output layer
     * @param netPath The path from which the neural network is being imported
     */
    public NeuralNetwork(File mixedData, int rngseed, int height, int width, int channels, int batchSize, int outputNum, String netPath) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        init();
        log.info("Building Neural Network from import...");
        loadNet(netPath);
    }

    /**
     * Constructor for preemptively defined training and testing data.
     * Also for if need to create a neural network.
     *
     * @param trainData Path to file with training data
     * @param testData Path to file with
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize
     * @param outputNum The number of nodes in the output layer
     */
    public NeuralNetwork(File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        init();
        log.info("Building Neural Network from scratch...");
        buildNet();
    }

    /**
     * Constructor for preemptively defined training and testing data.
     * Also for if importing an already built neural network.
     *
     * @param trainData Path to file with training data
     * @param testData Path to file with
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize
     * @param outputNum The number of nodes in the output layer
     * @param netPath The path from which the neural network is being imported
     */
    public NeuralNetwork(File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum, String netPath) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        init();
        log.info("Building Neural Network from import...");
        loadNet(netPath);
    }

    private void init(){
        this.rngseed = rngseed;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.outputNum = outputNum;
        this.netPath = netPath;
        ranNumGen = new Random(rngseed);

        JsonPathLabelGenerator label = new JsonPathLabelGenerator();
        recordReader = new JsonImageRecordReader(height, width, channels, label);
//        recordReader.setListeners(new LogRecordListener());
    }

    /**
     * Builds a neural network using gradient descent with regularization algorithm.
     */
    private void buildNet() {
        int layer1 = 100;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006) //alpha
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height*width*channels)
                        .nOut(layer1)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(layer1)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
    }

    private void loadNet(String NetPath) throws IOException {
        File locationToSave = new File(NetPath);
        model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    }

    /**
     * Trains the neural network with the training data.
     *
     * @param numEpochs Determines the number of times the model iterates through the training data set
     */
    public void train(int numEpochs) throws IOException {
        //if trainingReady is true and evaluatingReady is false
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, this.ranNumGen);
        recordReaderInit(train);

        //Displays how well neural network is training
//        model.setListeners(new ScoreIterationListener(10));

        for(int i = 0; i < numEpochs; i++) {
            model.fit(iter);
        }
    }

    /**
     * Evaluates the neural network by running the network through the testing data set.
     *
     * @returns eval The output of the evaluation
     */
    public Evaluation evaluate() throws IOException {
        //if trainingReady is false and evaluatingReady is true
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);
        recordReaderInit(test);

        Evaluation eval = new Evaluation(outputNum);

        while(iter.hasNext()) {
            DataSet next = iter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }
        return eval;
    }

    /**
     * Creates a record reader.
     *
     * @param file Name of path to the database wanting to be initialized
     */
    private void recordReaderInit(FileSplit file) throws IOException {
//        recordReader.reset();
        recordReader.initialize(file);
        iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        normalizeData();
    }

    /**
     * Saves the trained neural network
     *
     * @param filepath The path and file to which the neural network should be save to
     */
    public void saveBuild(String filepath) throws IOException {
        File saveLocation = new File(filepath);
        boolean saveUpdater = false; //want to enable retraining of data
        ModelSerializer.writeModel(model,saveLocation,saveUpdater);
    }
    
    /**
     * For testing purposes. Displays images with labels from a given database.
     *
     * @param numImages The number of images to display
     */
    public void imageToLabelDisplay(int numImages){
        for (int i = 0; i < numImages; i++) {
            DataSet ds = iter.next();
            System.out.println(ds);
            System.out.println(iter.getLabels());
        }
    }

    /** Normalizes the data to a value between 0 and 1 */
    private void normalizeData(){
        //Normalize pixel data
        scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
    }
}
