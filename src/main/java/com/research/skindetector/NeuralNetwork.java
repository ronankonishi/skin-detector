package com.research.skindetector;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Random;

/**
 * Convolutional Neural Network class that applies Supervised Learning for Skin Cancer Detection.
 *
 * This is the main class that builds, trains, and evaluates the neural network.
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
    DataSetIterator test_iter, train_iter;
    int numEpochs = 1; //number of times trained through dataset

    /**
     * Constructor for non-distinguished training and testing data (have all data in single directory, mixed data).
     * Also used if needing to build a neural network.
     * Note that you still must call the .build() function in the main class
     *
     * @param mixedData Path to file with mixed data
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize The number of images in a given minibatch
     * @param outputNum The number of nodes in the output layer
     * @throws IOException
     */
    public NeuralNetwork(File mixedData, File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        this.rngseed = rngseed;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.outputNum = outputNum;
        this.netPath = netPath;
        ranNumGen = new Random(rngseed);
        vectorization();
        dataSplitter(mixedData, trainData, testData);
        log.info("Building Neural Network from scratch...");
    }

    /**
     * Constructor for non-distinguished training and testing data (have all data in single directory, mixed data).
     * Also used if importing an already built neural network.
     *
     * @param mixedData Path to file with mixed data
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize The number of images in a given minibatch
     * @param outputNum The number of nodes in the output layer
     * @param netPath The path from which the neural network is being imported
     * @throws IOException
     */
    public NeuralNetwork(File mixedData, File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum, String netPath) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        this.rngseed = rngseed;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.outputNum = outputNum;
        this.netPath = netPath;
        ranNumGen = new Random(rngseed);
        vectorization();
        dataSplitter(mixedData, trainData, testData);
        log.info("Building Neural Network from import...");
        loadNet(netPath);
    }

    /**
     * Constructor for preemptively defined training and testing data.
     * Also for if needing to create a neural network.
     * Note that you still must call the .build() function in the main class
     *
     * @param trainData Path to file with training data
     * @param testData Path to file with
     * @param rngseed Integer that allows for constant random generated value
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for gray scaled and 3 for RGB)
     * @param batchSize The number of images in a given minibatch
     * @param outputNum The number of nodes in the output layer
     * @throws IOException
     */
    public NeuralNetwork(File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        this.rngseed = rngseed;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.outputNum = outputNum;
        this.netPath = netPath;
        ranNumGen = new Random(rngseed);
        vectorization();
        log.info("Building Neural Network from scratch...");
//        buildNet();
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
     * @param batchSize The number of images in a given minibatch
     * @param outputNum The number of nodes in the output layer
     * @param netPath The path from which the neural network is being imported
     * @throws IOException
     */
    public NeuralNetwork(File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum, String netPath) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        this.rngseed = rngseed;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.outputNum = outputNum;
        this.netPath = netPath;
        ranNumGen = new Random(rngseed);
        vectorization();
        log.info("Building Neural Network from import...");
        loadNet(netPath);
    }

    /**
     * Performs Image Preprocessing by labeling all of the data, scaling the images to a uniform,
     * and vectorizing the pixels values.
     */
    private void vectorization(){
        JsonPathLabelGenerator label = new JsonPathLabelGenerator();
        recordReader = new JsonImageRecordReader(height, width, channels, label);
//        recordReader.setListeners(new LogRecordListener()); //uncomment to check the label for each input data
    }

    /**
     * Builds a neural network using gradient descent with regularization algorithm.
     *
     * @param learningRate The rate at which the neural network learns
     * @param momentum The rate for increasing training rate (note that it was felt unused with Adam Updater)
     * @param weightDecay
     */
    public MultiLayerNetwork buildNet(double learningRate, double momentum, double weightDecay) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed) //saves this neural netwokr wiht the given optimizations
                .iterations(1) // Training iterations as above
                .regularization(true).l2(weightDecay) //prevents overfitting (REVIEW THIS)
                .learningRate(learningRate) // alpha from gradient descent (how fast it goes down the gradient)
                .weightInit(WeightInit.RELU) //method of randomizing weights in a gaussian distribution with equal variance throughout each layer
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)// (ideal for big data and big models) uses randomized minibatches to compute cost and gradient descent, which iterates through many minibatches
                .updater(new Adam()) //helps remove oscillation, by rounding off. (helps to start with minimal momentum 0.5 and after following gradient path well, increase momentum)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5) //5x5 pixel feature, stride moves
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels) //3
                        .stride(1, 1) //1px to right, then 1 px down
                        .nOut(20)
                        .activation(Activation.LEAKYRELU) //https://www.youtube.com/watch?v=-7scQpJT7uo (REVIEW) (shrinks values (0-256 for R in RGB) from a value between x and 1 (where 0 to 1 is linear with slope of 1 and x to 0 is linear with minimal slope)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // takes max in 2x2 pixel and represents a new image where that 2x2 space equals a single pixel
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height,width,channels))
                .backprop(true).pretrain(false).build();

        model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

    /**
     * Gets the neural network model
     *
     * @return model
     */
    public MultiLayerNetwork getNet(){
        return model;
    }

    /**
     * Enables the User Interface to allow for analysis of the neural network training.
     */
    public void UIenable(){
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage(); //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 10;
        this.getNet().setListeners(new StatsListener(statsStorage, listenerFrequency));

        uiServer.attach(statsStorage);
        for(int i = 0; i < numEpochs; i++) {
            this.getNet().fit(this.getTrainIter());
        }
    }

    /**
     * Trains the neural network with the training data.
     *
     * @throws IOException
     */
    public void train() throws IOException {
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, this.ranNumGen);

        recordReader.initialize(train);
        DataSetIterator temp_iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler = new ImagePreProcessingScaler(0,1); //normalization
        scaler.fit(temp_iter);
        temp_iter.setPreProcessor(scaler);
        train_iter = new AsyncDataSetIterator(temp_iter);

//        model.setListeners(new ScoreIterationListener(10)); //Uncomment to display how well the neural network is training

        //disables the java garbage collector
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        Nd4j.getMemoryManager().togglePeriodicGc(false);

        //Accommodates for java memory issues by optimizing usage of computing powers
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                .prefetchBuffer(8)
                .workers(2)
                .averagingFrequency(5)
                .reportScoreAfterAveraging(false)
                .workspaceMode(WorkspaceMode.SEPARATE)
                .build();
    }

    /**
     * Evaluates the neural network by running the network through the testing data set.
     *
     * @return eval The output of the evaluation
     * @throws IOException
     */
    public Evaluation evaluate() throws IOException {
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);

        recordReader.initialize(test);
        test_iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(test_iter);
        test_iter.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(outputNum);

        ROC roceval = new ROC(outputNum);
        roceval.calculateAUC();
        roceval.calculateAUCPR();
        roceval.getRocCurve();
        roceval.getPrecisionRecallCurve();
        EvaluationTools.exportRocChartsToHtmlFile(roceval, new File("roc_chart_HAM.html"));
//        model.doEvaluation(iteratorTest, eval, roceval);

        while(test_iter.hasNext()) {
            DataSet next = test_iter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }
        return eval;
    }

    /**
     * Saves the trained neural network
     *
     * @param filepath The path and file to which the neural network should be save to
     * @throws IOException
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
    public void imageToLabelDisplay(int numImages, DataSetIterator iter){
        for (int i = 0; i < numImages; i++) {
            DataSet ds = iter.next();
            System.out.println(ds);
            System.out.println(iter.getLabels());
        }
    }

    /**
     *
     * @param mixedDataset The path to the dataset with all of the data
     * @param trainData The path to which the training dataset should be placed
     * @param testData The path to which the testing dataset should be placed
     * @throws IOException
     */
    private void dataSplitter(File mixedDataset, File trainData, File testData) throws IOException {
        this.trainData = trainData;
        this.testData = testData;

        File[] mixedData = mixedDataset.listFiles();
        String temp1, temp2;
        for(int i = 0; i < mixedData.length/2; i++){
            double random = Math.random();
            temp1 = mixedData[i*2].toString();
            temp2 = mixedData[i*2+1].toString();

            if (random > 0.25) {
                Files.move(mixedData[i*2].toPath(), new File(trainData + "\\" + temp1.substring(temp1.lastIndexOf('\\')+1)).toPath());
                Files.move(mixedData[i*2+1].toPath(), new File(trainData + "\\" + temp2.substring(temp2.lastIndexOf('\\')+1)).toPath());
            } else {
                Files.move(mixedData[i*2].toPath(), new File(testData + "\\" + temp1.substring(temp1.lastIndexOf('\\')+1)).toPath());
                Files.move(mixedData[i*2+1].toPath(), new File(testData + "\\" + temp2.substring(temp2.lastIndexOf('\\')+1)).toPath());
            }
        }
    }

    /**
     * Loads the neural network
     *
     * @param NetPath The path to import the neural network from (should be .zip format)
     * @throws IOException
     */
    private void loadNet(String NetPath) throws IOException {
        File locationToSave = new File(NetPath);
        model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
    }

    /**
     * Gets the trained iterator
     *
     * @return train_iter
     */
    public DataSetIterator getTrainIter(){
        return train_iter;
    }
}