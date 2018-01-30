package com.research.skindetector;

import org.datavec.api.conf.Configuration;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.api.util.files.URIUtil;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URI;
import java.util.*;

public class TestInputSpliter {

//    InputSplit inputSplit;
//    protected boolean appendLabel = false;
//    protected PathLabelGenerator labelGenerator = null;

    protected Iterator<File> iter;
    protected Configuration conf;
    protected File currentFile;
    protected PathLabelGenerator labelGenerator = null;
    protected List<String> labels = new ArrayList<>();
    protected boolean appendLabel = false;
    protected boolean writeLabel = false;
    protected List<Writable> record;
    protected boolean hitImage = false;
    protected int height = 28, width = 28, channels = 1;
    protected boolean cropImage = false;
    protected ImageTransform imageTransform;
    protected BaseImageLoader imageLoader;
    protected InputSplit inputSplit;
    protected Map<String, String> fileNameMap = new LinkedHashMap<>();
    protected String pattern; // Pattern to split and segment file name, pass in regex
    protected int patternPosition = 0;

//    public final static String HEIGHT = NAME_SPACE + ".height";
//    public final static String WIDTH = NAME_SPACE + ".width";
//    public final static String CHANNELS = NAME_SPACE + ".channels";
//    public final static String CROP_IMAGE = NAME_SPACE + ".cropimage";
//    public final static String IMAGE_LOADER = NAME_SPACE + ".imageloader";

    public void initialize(InputSplit split) throws IOException {
//        if (imageLoader == null) {
//            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
//        }
        inputSplit = split;
        URI[] locations = split.locations();
        if (locations != null && locations.length >= 1) {
            if (appendLabel && labelGenerator != null) {
                Set<String> labelsSet = new HashSet<>();
                for (URI location : locations) {
                    File imgFile = new File(location);
                    File parentDir = imgFile.getParentFile();
                    String name = labelGenerator.getLabelForPath(location).toString();
                    labelsSet.add(name);
                    if (pattern != null) {
                        String label = name.split(pattern)[patternPosition];
                        fileNameMap.put(imgFile.toString(), label);
                    }
                }
                labels.clear();
                labels.addAll(labelsSet);
            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
        } else
            throw new IllegalArgumentException("No path locations found in the split.");

        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        Collections.sort(labels);
    }

}
