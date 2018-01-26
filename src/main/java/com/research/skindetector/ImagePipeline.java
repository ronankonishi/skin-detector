package com.research.skindetector;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
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
        int height = 1000;
        int width = 1000;
        int channels = 3; //RGB
        int rngseed = 11;
        Random ranNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 5;
        int numEpochs = 15;

        File testing = new File("/Users/Ronan/Desktop/Testing/");
        //Define File Paths
        File trainData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-1");
//        File testData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-2");

        //create randomized data
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);
//        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);

        //creates label
//        try {
//            JsonReader jsonReader = Json.createReader(new FileReader("ISIC_0011403.json"));
//            JsonObject json = jsonReader.readObject();
//            String ID = json.getString("_id");
//            String malignancy = json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant");
//        }
//        catch (FileNotFoundException e){ e.printStackTrace();}
//        catch (IOException e){ e.printStackTrace();}
////		catch (ParseException e){ e.printStackTrace();}
//        catch (Exception e){ e.printStackTrace();}

        //rescale, converts, and labels images
        BaseImageRecordReader recordReader = new BaseImageRecordReader(height, width, channels, label);
        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        //normalize pixel data
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

// Build Our Neural Network

        log.info("**** Build Model ****");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        // The Score iteration Listener will log
        // output to show how well the network is training
        model.setListeners(new ScoreIterationListener(10));

        log.info("*****TRAIN MODEL********");
        for(int i = 0; i < numEpochs; i++){
            model.fit(dataIter);
        }
    }

    private static class LabelGen implements PathLabelGenerator {

        public Writable getLabelForPath(String path) {
            if (path.endsWith("0.txt"))
                return new IntWritable(0);
            else if (path.endsWith("1.txt"))
                return new IntWritable(1);
            else
                return new IntWritable(2);
        }

        public Writable getLabelForPath(URI uri) {
            return getLabelForPath(uri.getPath());
        }

        public boolean inferLabelClasses() {
            return true;
        }
    }
}

