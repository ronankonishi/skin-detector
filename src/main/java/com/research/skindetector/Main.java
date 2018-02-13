package com.research.skindetector;

import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
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
    
    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {
        //Image Specifications
        int height = 80; //px height
        int width = 80; //px width
        int channels = 3; //RGB
        int rngseed = 11;
        int batchSize = 500;
        int outputNum = 2;
        int numEpochs = 10;

        File trainData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-1");
        File testData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-2");

        NeuralNetwork network = new NeuralNetwork(trainData, testData, rngseed, height, width, channels, batchSize, outputNum);

        log.info("**** Build Model ****");
        network.build();

        log.info("*****TRAIN MODEL********");
        network.train(numEpochs);

        log.info("*****SAVE TRAINED MODEL******");
        network.saveBuild();

        log.info("*****EVALUATE MODEL*******");
        log.info(network.evaluate().stats());
    }
}
