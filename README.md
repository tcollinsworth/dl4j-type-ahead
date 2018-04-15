# RNN Classification using dl4j

Upload a zip of directories whose names are the classifications.

The files in each directory are text examples.

The app will iterate the directories converting paragraphs to files whose names are the labels.
All files will be placed in a /src/main/resources/examples/train dir.
As each directory is completed, it will split the examples into test and validation directories 50/25/25%.

As documents are iterated, it accumulates count, max size, unique characters, maxCharacter val for all label examples.
At the end it prints stats and creates a character map file that can be used to vectorize all unique chars.

## Book Classifications

https://www.gutenberg.org/catalog/

## Language

https://mashable.com/2013/07/11/lorem-ipsum/#WlGh0w0wAiq3

Use google translations to create example documents from anything, i.e., 

https://www.google.com/search?q=translation&oq=translation


# Features - WIP

  * Train
  * Evaluate
  * load/save/continue training models
  * upload/manage raw examples data
  * Infer/predict classifications of new data
