package com.research.skindetector;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.net.URI;
public class JsonPathLabelGenerator implements PathLabelGenerator {

    public JsonPathLabelGenerator() {}

    @Override
    public Writable getLabelForPath(String path) {
        return
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return
    }

    @Override
    public boolean inferLabelClasses(){
        return true;
    }

}
