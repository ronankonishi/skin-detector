package com.research.skindetector;

import org.apache.commons.io.FilenameUtils;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Writable;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;
import java.io.File;
import java.net.URI;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.datavec.api.writable.Text;

/**
 * JSON Path Label Generator
 *
 * Label generator for a given JPG image, extracted from a Json file
 *
 * @author Ronan Konishi
 * @version 1.0
 */
public class JsonPathLabelGenerator implements PathLabelGenerator {

    /** Constructor */
    public JsonPathLabelGenerator() {}

    /**
     * Gets the JSON path for a given file JPG image.
     * Note that the comments make testing and debugging the method much simpler.
     *
     * @param JpgPath The path of the JPG image
     * @return benign or malignant The status of the given input JPG image written in the JSON file
     * @return null Returns null if the method fails to create a JSON Reader object
     */
    @Override
    public Writable getLabelForPath(String JpgPath) {
//        System.out.println(System.getProperty("user.dir")); // read working directory
//        String file = "C:\\Users\\Ronan\\ISIC-images\\ISIC-images\\UDA-1\\ISIC_XXXXXXX.json"; //temporary read file
        String JsonPath = fileExtensionRename(JpgPath, "json");
//        System.out.println(JsonPath); //Current Json Path print
        try {
            JsonReader jsonReader = Json.createReader(new FileReader(JsonPath)); //Jsonath is absolute file path
//            JsonReader jsonReader = Json.createReader(new FileReader(file)); //file is temporary read file
            JsonObject json = jsonReader.readObject();
//            System.out.println(new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant")));
            return new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant"));
        } catch (FileNotFoundException e){ e.printStackTrace();}
       catch (IOException e){ e.printStackTrace();}
       catch (Exception e){ e.printStackTrace();}
       return null;
    }

    /**
     * Gets the JSON path for a given file JPG image.
     * 
     * @param uri The path of the JPG image
     * @return benign or malignant The status of the given input JPG image written in the JSON file
     * @return null Returns null if the method fails to create a JSON Reader object
     */
    @Override
    public Writable getLabelForPath(URI uri){
        return getLabelForPath(new File(uri).toString()); //remove getName() for absolute path
    }

    /**
     * Renames the given input file extension.
     *
     * @param input File to rename
     * @param newExtension The new extension to give to file
     * @return The output file with new extension
     */
    private static String fileExtensionRename(String input, String newExtension) {
        if (oldExtension.equals("")) {
            return input + "." + newExtension;
        } else {
            return input.replaceFirst(Pattern.quote("." + getFileExtension(input)) + "$", Matcher.quoteReplacement("." + newExtension));
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

    /**
     * Only compatible with newer versions of deeplearning4j.
     */
//    @Override
//    public boolean inferLabelClasses(){
//        return true;
//    }
}
