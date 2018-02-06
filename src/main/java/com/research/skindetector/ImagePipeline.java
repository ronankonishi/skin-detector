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

public class ImagePipeline {
    private static Logger log = LoggerFactory.getLogger(ImagePipeline.class);

    public static void main(String[] args) throws IOException {
        //Image Specifications
        int height = 80; //px height
        int width = 80; //px width
        int channels = 3; //RGB
        int rngseed = 11;
        Random ranNumGen = new Random(rngseed);
        int batchSize = 500;
        int outputNum = 2;
        int numEpochs = 10;

        //File Paths
        File trainData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-1");
        File testData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-2");

        //Create randomized data
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);

        //Retreive data labels
        JsonPathLabelGenerator label = new JsonPathLabelGenerator();

        //Rescale, convert, and label images
        JsonImageRecordReader recordReader = new JsonImageRecordReader(height, width, channels, label);
        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        //Normalize pixel data
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        //FOR TESTING PURPOSES: display 3 images with labels from database(set batchsize to 1)
        for (int i = 0; i < 3; i++){
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());
        }

        //Build Neural Network
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

        //Displays how well neural network is training
        model.setListeners(new ScoreIterationListener(10));

        //Train Neural Network
        log.info("*****TRAIN MODEL********");
        for(int i = 0; i < numEpochs; i++){
            model.fit(dataIter);
        }

        //Save Neural Network
        log.info("*****SAVE TRAINED MODEL******");
        File saveLocation = new File("trained_model.zip");

        boolean saveUpdater = false; //want to enable retraining of data

        ModelSerializer.writeModel(model,saveLocation,saveUpdater);


        //Evaluate Neural Network
        log.info("*****EVALUATE MODEL*******");
        recordReader.reset();

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        Evaluation eval = new Evaluation(outputNum);

        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(),output);
        }

        log.info(eval.stats());

    }
}

