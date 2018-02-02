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
public class JsonPathLabelGenerator implements PathLabelGenerator {

    public JsonPathLabelGenerator() {}

    @Override
    public Writable getLabelForPath(String JpgPath) {
//        System.out.println(System.getProperty("user.dir")); // read working directory
//        String file = "C:\\Users\\Ronan\\ISIC-images\\ISIC-images\\UDA-1\\ISIC_XXXXXXX.json"; //temporary read file
        String JsonPath = renameFileExtension(JpgPath, "json");
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

    @Override
    public Writable getLabelForPath(URI uri){
        return getLabelForPath(new File(uri).toString()); //remove getName() for absolute path
    }

    private static String renameFileExtension(String source, String newExtension) {
        String target;
        String currentExtension = getFileExtension(source);

        if (currentExtension.equals("")) {
            target = source + "." + newExtension;
        } else {
            target = source.replaceFirst(Pattern.quote("." + currentExtension) + "$", Matcher.quoteReplacement("." + newExtension));
        }
        return target;
    }

    private static String getFileExtension(String f) {
        String ext = "";
        int i = f.lastIndexOf('.');

        if (i > 0 &&  i < f.length() - 1) {
            ext = f.substring(i + 1);
        }
        return ext;
    }

//    @Override
//    public boolean inferLabelClasses(){
//        return true;
//    }
}
