package com.research.skindetector;

import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.research.skindetector.utilities.DataUtilities;

import java.io.File;
import java.io.IOException;

/**
 * Inspired by tom hanlon (a developer from deeplearning4j)
 */

public class DataDownloader {
    private static Logger log = LoggerFactory.getLogger(Main.class);
    public static String DATA_URL = "http://github.com/RonanK687/ISIC_Dataset/raw/master/ISIC_Dataset.tar.gz";
    public static String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "ISIC_Dataset/");

    public DataDownloader(){}

    public DataDownloader(String DATA_URL, String DATA_PATH){
        this.DATA_URL = DATA_URL;
        this.DATA_PATH = DATA_PATH;
    }

    public void download() throws IOException {
        File directory = new File(DATA_PATH);

        if(!directory.exists()){
            directory.mkdir();
        }

        String filePath = "/ISIC_Dataset.tar.gz";
        String archivePath = DATA_PATH + filePath;
        File archiveFile = new File(archivePath);
        File extractedFile = new File(DATA_PATH + "ISIC_Dataset");

        if (!archiveFile.exists()){
            log.info("Downloading Database. Please wait...");

            String tmpDirStr = System.getProperty("java.io.tmpdir");
            String archizePath = DATA_PATH + filePath;

            if (tmpDirStr == null) {
                throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir");
            }

            File f = new File(archizePath);
            if (!f.exists()) {
                DataUtilities.downloadFile(DATA_URL, archizePath);
                log.info("Data downloaded to ", archizePath);
            } else {
                log.info("Using existing directory at ", f.getAbsolutePath());
            }

            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archivePath, DATA_PATH);
        } else {
            log.info("Data (.tar.gz file) already exists at {}", archiveFile.getAbsolutePath());
            if (!extractedFile.exists()) {
                //Extract tar.gz file to output directory
                DataUtilities.extractTarGz(archivePath, DATA_PATH);
            } else {
                log.info("Data (extracted) already exists at {}", extractedFile.getAbsolutePath());
            }
        }
    }

    public String getDataURL(){
        return DATA_URL;
    }

    public void setDataURL(String DATA_URL){
        this.DATA_URL = DATA_URL;
    }

    public String getDataPath(){
        return DATA_PATH;
    }

    public void setDataPath(String DATA_PATH){
        this.DATA_PATH = DATA_PATH;
    }

}
