package com.research.skindetector;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.utilities.DataUtilities;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

/**
 * Inspired by tom hanlon, who posted his code on deeplearning4j
 * 
 * 
 */

public class DataDownloader {
//   public static final String DATA_URL = "http://github.com/RonanK687/ISIC_Dataset/raw/master/ISIC_Dataset.tar.gz";

//   public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "ISIC_Dataset/");

  public DataDownloader(String DATA_URL, String DATA_PATH){
    this.DATA_URL = DATA_URL;
    this.DATA_Path = DATA_PATH;
  }
  
  private static void download() {
    File directory = new File(DATA_PATH);
    if(!directory.exists()){
      directory.mkdir();
    }
    
    File archiveFile = new File(archizePath);
    
    
    
  }

  
  
}
