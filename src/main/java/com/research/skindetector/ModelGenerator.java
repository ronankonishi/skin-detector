//package com.research.skindetector;
//
//import org.datavec.api.records.reader.RecordReader;
//import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
//import org.datavec.api.split.FileSplit;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.slf4j.LoggerFactory;
//
//import java.io.File;
//import java.io.IOException;
//
//public class ModelGenerator {
//
//    private static final long SEED = 11;
//    private static final int HEIGHT = 32;
//    private static final int WIDTH = 32;
//    private static final int NUM_CHANNELS = 1;
//    private static final int NUM_LABELS = 29;
//    private static final int BATCH_SIZE = 100;
//    private static final int ITERATIONS = 1;
//    private static final int LABEL_INDEX = 1024;
//    private static final String PATH_TO_TRAINING_DATA = "";
//    private static final String PATH_TO_TESTING_DATA = "";
//
//    private static final org.slf4j.Logger log = LoggerFactory.getLogger(ModelGenerator.class);
//
//    private static DataSetIterator readCSVDataset(String csvFileClasspath, int BATCH_SIZE, int LABEL_INDEX, int numClasses)
//            throws IOException, InterruptedException {
//
//        RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new File(csvFileClasspath)));
//        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, BATCH_SIZE, LABEL_INDEX, numClasses);
//
//        return iterator;
//    }
//
//    public static void main(String[] args) {
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                        .updater(Updater.NESTEROVS).momentum(0.9)
//                        .learningRate(learningRate)
//                        .list(
//                                new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes).activation("relu").build(),
//                                new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(numHiddenNodes).nOut(numOutputs).build()
//                        ).backprop(true).build();
//    }
//}
