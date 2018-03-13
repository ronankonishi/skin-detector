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
    static int rngseed;
    static Random ranNumGen;
    static int outputNum;
    static JsonImageRecordReader recordReader;
    static int height, width, channels;

    public static void main(String[] args) throws IOException {
        rngseed = 11;
        int batchSize = 1000;
        outputNum = 2;
        height = 50;
        width = 50;
        channels = 3;
        //Get our network and training data

        File trainData = new File("C:\\Users\\ronan\\Desktop\\testsmall\\trainData\\");

        ranNumGen = new Random(rngseed);

        MultiLayerNetwork net = getMnistNetwork();
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);

        System.out.println(trainData.toPath());
        System.out.println(ranNumGen);
        System.out.println(train.toString());

        JsonPathLabelGenerator label = new JsonPathLabelGenerator();
        recordReader = new JsonImageRecordReader(height, width, channels, label);
        recordReader.initialize(train);
        DataSetIterator temp_iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(temp_iter);
        temp_iter.setPreProcessor(scaler);
//        iter = new AsyncDataSetIterator(temp_iter);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Start training:
        net.fit(temp_iter);

        //Finally: open your browser and go to http://localhost:9000/train
    }
    public static MultiLayerNetwork getMnistNetwork(){

//        int nChannels = 1; // Number of input channels
//        int outputNum = 10; // The number of possible outcomes
//        int iterations = 1; // Number of training iterations
//        int seed = 123; //
//        int numEpochs = 1; //number of iterations through entire dataset

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .iterations(1) // Training iterations as above
                .regularization(true).l2(0.0005)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
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

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }
//
//    public static DataSetIterator getMnistData(){
//        try{
//            MnistDownloader.download(); //Workaround for download location change since 0.9.1 release
//            return new MnistDataSetIterator(64,true,12345);
//        }catch (IOException e){
//            throw new RuntimeException(e);
//        }
//    }

}


//    private static Logger log = LoggerFactory.getLogger(Main.class);
//
//    public static void main(String[] args) throws IOException {
//        //Image Specifications
//        int height = 50; //px height
//        int width = 50; //px width
//        int channels = 3; //RGB
//        int rngseed = 11;
//        int batchSize = 1000;
//        int outputNum = 2;
//        int numEpochs = 1; //number of iterations through entire dataset
//
////        File trainData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-1");
////        File testData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-2");
//
////        DataDownloader dataSet = new DataDownloader();
////        dataSet.download();
////        File UnpackagedISICData = new File(dataSet.getDataPath()); //unsure still NEEDS TESTING
////
////        NeuralNetwork network = new NeuralNetwork(UnpackagedISICData, rngseed, height, width, channels, batchSize, outputNum);
//
////        File mixedData = new File("/Users/Ronan/Desktop/ISIC_Dataset");
//
//        File mixedData = new File("C:\\Users\\ronan\\Desktop\\testsmall\\mixedData\\");
//        File trainData = new File("C:\\Users\\ronan\\Desktop\\testsmall\\trainData\\");
//        File testData = new File("C:\\Users\\ronan\\Desktop\\testsmall\\testData\\");
//
//        NeuralNetwork network = new NeuralNetwork(mixedData, trainData, testData, rngseed, height, width, channels, batchSize, outputNum);
//
//        log.info("*****TRAIN MODEL********");
//        network.train(numEpochs);
//
////        log.info("*****SAVE TRAINED MODEL******");
////        network.saveBuild("trained_model.zip");
//
////        log.info("*****EVALUATE MODEL*******");
////        log.info(network.evaluate().stats());
//
//        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
//    }
//}
