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
import java.nio.file.Files;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * JSON Image Record Reader
 *
 * A formated dataset for JPG images and labels within a JSON file.
 *
 * @author Ronan Konishi
 * @version 1.0
 */
public class JsonImageRecordReader extends BaseImageRecordReader {

    /**
     * Constructor
     *
     * @param height The height of image in pixels
     * @param width The width of image in pixels
     * @param channels The number of channels (e.g. 1 for grayscaled and 3 for RGB)
     * @param labelGenerator Label (of either malignant or benign) for a given JPG image
     */
    public JsonImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator){
        super(height, width, channels, labelGenerator);
    }

    /**
     * Initializes the image record reader by transforming input images and labeling them with their corresponding 
     * status (either malignant or benign) in a format compatible by the deeplearning4j library.
     * 
     * @param split The file path for the dataset that should be initialized
     */
    @Override
    public void initialize(InputSplit split) throws IOException {
        //transforms image to the given height and width for the given channel
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
                    File imgFile = new File(location);
//                    String name = labelGenerator.getLabelForPath(location).toString();
//                    labelsSet.add(name);
//                    if (pattern != null) {
//                        String label = name.split(pattern)[patternPosition];
//                        fileNameMap.put(imgFile.toString(), label);
//                    }
                    if(labelGenerator.getLabelForPath(location) != null) {
                        String name = labelGenerator.getLabelForPath(location).toString();
                        labelsSet.add(name);
                    } else {
                        File garbageCollect = new File("C:\\Users\\ronan\\Desktop\\test\\garbageCollect\\");
                        if (!garbageCollect.exists()) {
                            garbageCollect.mkdir();
                        }
                        File tempjson = new File((fileExtensionRename(imgFile.toString(),"json")));
//                        System.out.println(imgFile.toPath());
//                        System.out.println(garbageCollect.toPath()  + "\\" + imgFile.toString().substring(imgFile.toString().lastIndexOf('\\')+1));
                        Files.move(imgFile.toPath(), new File(garbageCollect.toPath()  + "\\" + imgFile.toString().substring(imgFile.toString().lastIndexOf('\\')+1)).toPath());
//                        Files.move(tempjson.toPath(), new File(garbageCollect.toPath()  + "\\" + tempjson.toString().substring(tempjson.toString().lastIndexOf('\\')+1)).toPath());
                    }
                }
                labels.clear();
                System.out.println("clear");
                labels.addAll(labelsSet);
                System.out.println("addAll labelsSet");
            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
            System.out.println("Randomization");
        } else
            throw new IllegalArgumentException("No path locations found in the split.");

        if (split instanceof FileSplit) {
            //remove the root directory
            System.out.println("Remove root directory?");
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        Collections.sort(labels);
        System.out.println("Collection sorted");
    }
    private static String fileExtensionRename(String input, String newExtension) {
        String oldExtension = getFileExtension(input);

        if (oldExtension.equals("")) {
            return input + "." + newExtension;
        } else {
            return input.replaceFirst(Pattern.quote("." + oldExtension) + "$", Matcher.quoteReplacement("." + newExtension));
        }
    }

    /**
     * Gets the file extention.
     * @param input File to get extension from
     */
    private static String getFileExtension(String input) {
        int i = input.lastIndexOf('.');

        if (i > 0 &&  i < input.length() - 1) {
            return input.substring(i + 1);
        } else {
            return "";
        }
    }
}
