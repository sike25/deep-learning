# deep-learning

#### Data Exploration, Modeling, Regularization and Prediction (Image Recognition)
Using TensorFlow Keras API to explore the well-known CIFAR10 (Canadian Institute For Advanced Research) image dataset, building a simple feed-forward neural network, and evaluating the effects of playing with basic methods (activation functions, regularization, weight initialization, neuron and layer numbers etc.) to push accuracy past 53%.

#### Convolutional Neural Networks (Image Recognition)
Replacing the simple multi-layer perceptron with a convolutional neural network to classify the CIFAR10 (Canadian Institute For Advanced Research) image dataset. Trainable paramters must be kept under 200_000 and accuracy must surpass 72%.

#### Convolutional Neural Networks (Activity Recognition)
We train a CNN on the [UCI Human Activity Recognition](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) database. Thirty volunteers performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) to collect 3-axial linear acceleration and 3-axial angular velocity data. The nn design task focused on building narrow and deep Conv1D kernels starting from a fixed architecture with the goal of pushing accuracy past 90%. Options available were to adjust filters, kernel sizes and other hyperparameters. 

#### Retrieval Augmented Generation for Large Language Models
We combine [LLAMA 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) with the wikipedia pages of five 2024 movies and series (Damsel, Shogun, Baby Reindeer, Unfrosted and One Day) to teach it to reply correctly to prompts related to these documents, whose information postdate the LLM's training. We start by ensuring it answers incorrectly to the prompts before we set up the RAG, and correctly after.

#### Anomality Detection With Autoencoders
We create a fully-connected neural network based autoencoder in order to detect credit card frauds. We use this [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) where there are 492 labeled frauds and 284807 normal transactions by European cardholders in September 2013.

#### Authorship Attribution with BERT
Using the BERT (Bidirectional Encoder Representations from Transformers) to determine the authors of the unattributed Federalist Papers amongst Alexander Hamilton, James Madison, and John Jay. We loading, clean, tokenize and split the text data. After configuring BERT and the training process, we train the model and end with running inference.
