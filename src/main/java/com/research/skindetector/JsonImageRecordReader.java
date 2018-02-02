package com.research.skindetector;

import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.image.loader.NativeImageLoader;

import java.io.*;
import java.net.URI;
import java.util.*;

public class JsonImageRecordReader extends BaseImageRecordReader {

    public JsonImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator){
        super(height, width, channels, labelGenerator);
    }

    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
        inputSplit = split;
        URI[] locations = split.locations();
        if (locations != null && locations.length >= 1) {
            if (appendLabel && labelGenerator != null) {
//                if (appendLabel && labelGenerator != null && labelGenerator.inferLabelClasses()) {
                Set<String> labelsSet = new HashSet<>();
                for (URI location : locations) {
//                    System.out.print(location + "/n"); KEEP FOR DEBUGGING
                    File imgFile = new File(location);
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
