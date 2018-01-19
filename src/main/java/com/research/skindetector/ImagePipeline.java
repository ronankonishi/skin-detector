package com.research.skindetector;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
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

public class ImagePipeline {
    private static Logger log = LoggerFactory.getLogger(ImagePipeline.class);

    public static void main(String[] args) throws IOException {
        //Image Specifications
        int height = 1000;
        int width = 1000;
        int channels = 3;
        int rngseed = 11;
        Random ranNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 5;

        //Define File Paths
        File trainData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-1");
        File testData = new File("/Users/Ronan/ISIC-images/ISIC-images/UDA-2");


        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, ranNumGen);

        //parent directory label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        //rescale, converts, and labels images
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());


        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        //normalize pixel data
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        for(int i = 1; i < 3; i++){
            DataSet ds = dataIter.next();
            System.out.print(ds);
            System.out.println(dataIter.getLabels());
        }

    }

}
