# SignTheta - Two way Sign Language Translation
A web based application which accepts Audio/ Voice as input and converts it to corresponding Sign Language for Deaf people.
The interface works in two phases, 
* First, recognizes speech and converts it to text using JavaScript Web Speech API 
* Secondly, uses Machine based translation to translate English into ISL based grammar. Semi-structured parse tree of English text is modified to represent parse tree of ISL based on bi-linguistic rules.

![Screenshot](https://github.com/aanchal1308/MajorProject/blob/master/images/screenshot.png)
## Installation Guide

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
* ```Python >= 2.7```
* ```Browser supports Web Speech API```


### Installing
* Git clone repository
* Download all required packages for running python script server.py

### Running the code
* Run flask application server.py in localhost or server. Running server.py first time takes too long as it downloads 
Stanford-Parser over HTTP. Make sure you are connected to internet.
* Run index.html in browsers using php server ```php -S localhost:8080```




