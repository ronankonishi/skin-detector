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

//    public JsonPathLabelGenerator() {}
//
//        @Override
//        public Writable getLabelForPath(String path) {
//            // Label is in the directory
//            System.out.println(new File(path).getParent());
//            return new Text(FilenameUtils.getBaseName(new File(path).getParent()));
//        }
//
////        @Override
//        public Writable getLabelForPath(URI uri) {
////            System.out.println(uri);
//            return getLabelForPath(new File(uri).toString());
//        }

    @Override
    public Writable getLabelForPath(String JpgPath) {
//        System.out.println(getFileExtension(JpgPath));
//        System.out.println(System.getProperty("user.dir"));
//        String file = "C:\\Users\\Ronan\\ISIC-images\\ISIC-images\\UDA-1\\ISIC_0000000.json";
        String JsonPath = renameFileExtension(JpgPath, "json");
        System.out.println(JsonPath);
        try {
            JsonReader jsonReader = Json.createReader(new FileReader(JsonPath)); //path is file name
//            JsonReader jsonReader = Json.createReader(new FileReader(file)); //path is file name
            JsonObject json = jsonReader.readObject();
            System.out.println(new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant")));
//            return new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant"));
        } catch (FileNotFoundException e){ e.printStackTrace();}
       catch (IOException e){ e.printStackTrace();}
       catch (Exception e){ e.printStackTrace();}
       return null;
    }

    @Override
    public Writable getLabelForPath(URI uri){
        return getLabelForPath(new File(uri).toString()); //remove getName() for absolute path
    }

    public static String renameFileExtension
            (String source, String newExtension)
    {
        String target;
        String currentExtension = getFileExtension(source);

        if (currentExtension.equals("")){
            target = source + "." + newExtension;
        }
        else {
            target = source.replaceFirst(Pattern.quote("." +
                    currentExtension) + "$", Matcher.quoteReplacement("." + newExtension));

        }
        return target;
    }

    public static String getFileExtension(String f) {
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
//
//    /**
//     *
//     * @param path  the path to the file being read
//     * @return status of either benign or malignant as a Text
//     */
//    private Writable JsonExtraction(String path){
//       try {
//            JsonReader jsonReader = Json.createReader(new FileReader(path)); //path is file name
//            JsonObject json = jsonReader.readObject();
//            return new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant"));
//       }
//       catch (FileNotFoundException e){ e.printStackTrace();}
//       catch (IOException e){ e.printStackTrace();}
//       catch (Exception e){ e.printStackTrace();}
//       return null;
//    }
//
    //same as before but with URI
//    public Writable JsonExtraction(URI uri){
//        return getLabelForPath(new File(uri).toString());
//        try {
//            URI t1 = uri;
//            String t2 = uri.toString();
//            System.out.println(t2);
//            FileReader t3 = new FileReader((t2));
//            JsonReader jsonReader = Json.createReader(t3);
////            JsonReader jsonReader = Json.createReader(new FileReader((uri).toString())); //path is file name
//            JsonObject json = jsonReader.readObject();
//            return new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant"));
//        }
//        catch (FileNotFoundException e){ e.printStackTrace();}
//        catch (IOException e){ e.printStackTrace();}
//        catch (Exception e){ e.printStackTrace();}
//        return null;
//    }
}
