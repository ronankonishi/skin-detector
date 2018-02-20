# Convolutional Neural Networks for Skin Cancer Detection

At this current date, studies indicate promising results when applying artificial neural networks for image classification. However, the application to skin cancer detection is limited, and even moreso is the number of successful attempts via smartphones. This project was initiated in response to the low success rate of creating an efficient skin cancer diagnosing smartphone application, by applying a Convolution Neural Networks algorithm provided by the deeplearning4j library. This program allows for training, testing, and exporting a well-developed neural network.

## Current Status

* The program in its basic state is complete.
* The next step is to perform tests and optimize the neural network.
* I also plan to implement a script that automatically downloads the database of images used for training and testing the network. (isic-archive.com).
* I also plan to implement an automatic file distibutor, which will take the data from a single database and randomly sort the data into a training and testing file.
* Smartphone compatibiltiy is still under development. I plan to make it Android compatible first, and later expand to other operating systems.

## Compatibility
* Currently only tested on Windows Operating System
* Currently only tested on IntelliJ IDEA IDE

## Prerequisites

* [Java](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) - Programming Language
* [IntelliJ IDEA](https://www.jetbrains.com/idea/download/#section=windows) - IDE
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) - Allows for project download
* [Maven](https://maven.apache.org/download.cgi) - Dependency Management

## Installing 

1. Use the command line to enter the following:
```
git clone https://github.com/RonanK687/skin-detector.git
cd skin-detector
mvn clean install
```

2. Select Maven when building in IntelliJ and select SDK as jdk

## Author

* **Ronan Konishi**

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details. When using the images and annotations from the ISIC-archive, please abide by their [Terms Of Use](https://isic-archive.com/#termsOfUse).

## Acknowledgments

* Special thanks to The International Skin Imaging Collaboration (ISIC) for the open source database of skin cancer moles.
* Special thanks to the deeplearning4j team for their open source [artificial intelligence library](https://github.com/deeplearning4j).
