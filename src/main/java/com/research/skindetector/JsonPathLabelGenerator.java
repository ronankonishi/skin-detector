package com.research.skindetector;

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
import org.datavec.api.writable.Text;
public class JsonPathLabelGenerator implements PathLabelGenerator {

    public JsonPathLabelGenerator() {}

//    @Override
    public Writable getLabelForPath(String path) {
        return JsonExtraction(path);
    }

    public Writable getLabelForPath(URI uri){
        return JsonExtraction(uri);
    }
//
//    @Override
//    public Writable getLabelForPath(URI uri) {
//        return JsonExtraction(uri);
//    }

//    @Override
    public boolean inferLabelClasses(){
        return true;
    }

    /**
     *
     * @param path  the path to the file being read
     * @return status of either benign or malignant as a Text
     */
    private Writable JsonExtraction(String path){
       try {
            JsonReader jsonReader = Json.createReader(new FileReader(path)); //path is file name
            JsonObject json = jsonReader.readObject();
            return new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant"));
       }
       catch (FileNotFoundException e){ e.printStackTrace();}
       catch (IOException e){ e.printStackTrace();}
       catch (Exception e){ e.printStackTrace();}
       return null;
    }

    //same as before but with URI
//    public Writable JsonExtraction(URI uri){
//        try {
//            JsonReader jsonReader = Json.createReader(new FileReader((uri).toString())); //path is file name
//            JsonObject json = jsonReader.readObject();
//            return new Text(json.getJsonObject("meta").getJsonObject("clinical").getString("benign_malignant"));
//        }
//        catch (FileNotFoundException e){ e.printStackTrace();}
//        catch (IOException e){ e.printStackTrace();}
//        catch (Exception e){ e.printStackTrace();}
//        return null;
//    }
}
