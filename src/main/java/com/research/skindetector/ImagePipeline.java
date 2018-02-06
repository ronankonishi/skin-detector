package com.research.skindetector;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class ImagePipeline {
    private static Logger log = LoggerFactory.getLogger(ImagePipeline.class);

    public static void main(String[] args) throws IOException {
        //Image Specifications
        int height = 80; //px height
        int width = 80; //px width
        int channels = 3; //RGB
        int rngseed = 11;
        Random ranNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 2;
        int numEpochs = 5;

        //Define File Paths
        File trainData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-1");
//        File testData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-2");

        //create randomized data
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);
//        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);

        //gets labels
        JsonPathLabelGenerator label = new JsonPathLabelGenerator();

        //rescale, converts, and labels images
        JsonImageRecordReader recordReader = new JsonImageRecordReader(height, width, channels, label);
        recordReader.initialize(train);
//        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        //normalize pixel data
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        //TESTING, display 3 images with labels from database(set batchsize to 1)
//        for (int i = 0; i < 3; i++){
//            DataSet ds = dataIter.next();
//            System.out.println(ds);
//            System.out.println(dataIter.getLabels());
//        }

// Build Our Neural Network
//
        log.info("**** Build Model ****");

        int layer1 = 100;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
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

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

//        // The Score iteration Listener will log
//        // output to show how well the network is training
        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");
        for(int i = 0; i < numEpochs; i++){
            model.fit(dataIter);
        }
    }
}

