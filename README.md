# MultilayerPerceptron-ConvolutionalNetwork-and-CapsuleNetwork-for-Natural-Language-Processing-NLP
Multilayer Perceptron (MLP), Convolutional Neural Network (CNN) and Capsule Network (CapsNet) for Natural Language Processing (NLP)with Italian Messages

In this project I want to show you 3 different type of Neural Network in a NLP (Natural Language Processing) problem for Italian Messages. The problem is Hate Speech Detection.

########################## DATASET ##########################
The dataset of italian posts in this repository was given by EVALITA (http://www.evalita.it/) for the italian competition "HaSpeeDe" in 2018 (http://www.di.unito.it/~tutreeb/haspeede-evalita18/index.html).
The records are cleaned yet and the preprocessing phase is omitted because is very lengthy and depends from the preferences of each individual programmer.


####################### WORD2VEC MODEL #######################
In MultiLayer Perceptron and Convolutiona Neural Network models I use a Word2Vec model for the word-embedding and sentence-embedding.
In the real project I use a Word2Vec model trained with 200k posts downloaded from Facebook political pages but that file is too bigger for load it in github.
So I had to create a smaller Word2Vec model trained with only the dataset's records, but for this has a little information content. For this problem, if you try the networks you can't get a good value of accuracy for sure.
I recommend to create a personal Word2Vec model, training it how you want and with how many posts you want.
I post a script that can help you to create a Word2Vec model too.
