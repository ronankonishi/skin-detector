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

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class NeuralNetwork {

    File trainData, testData;
    int rngseed, height, width, channels, batchSize, outputNum;
    Random ranNumGen;
    MultiLayerNetwork model;
    JsonImageRecordReader recordReader;
    DataNormalization scaler;
    DataSetIterator iter;
    boolean testingReady = false;
    boolean evaluatingReady = false;

    public NeuralNetwork(File trainData, File testData, int rngseed, int height, int width, int channels, int batchSize, int outputNum) throws IOException {
        this.trainData = trainData;
        this.testData = testData;
        this.rngseed = rngseed;
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.outputNum = outputNum;

        ranNumGen = new Random(rngseed);

        JsonPathLabelGenerator label = new JsonPathLabelGenerator();

        //Rescale, convert, and label images
        recordReader = new JsonImageRecordReader(height, width, channels, label);
//        recordReader.setListeners(new LogRecordListener());

//        trainingReady = true;
//        testingReady = false;
    }

    public void build() {
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

        model = new MultiLayerNetwork(conf);
        model.init();
    }

//PUT CHECKS FOR TRAIN TO NOT OVERLAP
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

    private void recordReaderInit(FileSplit file) throws IOException {
//        recordReader.reset();
        recordReader.initialize(file);
        iter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        normalizeData();
    }

    public void saveBuild() throws IOException {
        File saveLocation = new File("trained_model.zip");
        boolean saveUpdater = false; //want to enable retraining of data
        ModelSerializer.writeModel(model,saveLocation,saveUpdater);
    }

    //FOR TESTING PURPOSES: display 3 images with labels from database(set batchsize to 1)
    public void imageToLabelDisplay(){
        for (int i = 0; i < 3; i++) {
            DataSet ds = iter.next();
            System.out.println(ds);
            System.out.println(iter.getLabels());
        }
    }

    private void normalizeData(){
        //Normalize pixel data
        scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
    }
}
