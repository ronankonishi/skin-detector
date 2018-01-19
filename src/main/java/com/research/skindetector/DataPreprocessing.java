package com.research.skindetector;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;

import java.util.Date;

/**
 *
 */
public class DataPreprocessing {
    public static void main(String[] args) {
        int LinesToSkip = 0;
        String delimiter = ",";

        String dataBaseDir = "/Users/Ronan/";
        String dataFileName = "skincancerData";
        String inputPath = dataBaseDir + dataFileName;
        String timeStamp = String.valueOf(new Date().getTime());
        String outputPath = dataBaseDir + "" + timeStamp;

//        Schema inputDataSchema = new Schema.Builder()
//                .build();

//        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
//                .build();



    }

}
