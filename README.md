# Causal Effect of Linguistic Register Level on Sentiment Classification Models

This project aims to investigate the causal effect of linguistic register level on sentiment classification models. The goal is to understand how the register level of text, which is one of the concepts that generate the text, influences the decisions made by sentiment analysis models based on language models like BERT.


## Introduction

With the rise of black box models like Deep Neural Networks (DNNs), it is crucial to understand what factors affect the model's decisions. Language models, like all data-based models, may contain unwanted biases from the data. This project aims to estimate the causal effect of the register level concept on language models such as BERT.

## Methodology

The project utilizes the concept of causal inference to measure the causal effect of register level on sentiment classification models. The do-operator is simulated by replacing adjectives in the text with their synonyms characterized by higher or lower register levels. This allows for the measurement of the causal concept effect of register level on language models in tasks like sentiment analysis.

The project uses a dataset of classified words and their register levels, as well as a database of synonyms, to perform the text transformations. The transformed texts are then used to calculate the Causal Concept Effect (CaCE) on the sentiment classification model.

## Dataset

The project uses the following datasets:

- IMDB Review Dataset: Contains 50,000 movie reviews with ratings on a 1-5 scale. Ratings of 4 and above are considered positive, while the rest are considered negative.
- Tweets Complaint Dataset: Contains 3,200 tweets, half of which are complaints about air companies and the other half are not.

## Experiments

The experiments utilize the uncased BERT-base pre-trained text representation model for sentiment classification. The data is divided into a train set and a test set with a ratio of 9:1. The model is trained on the train set and the CaCE is calculated on the test set before and after the text transformation.

## Results

The results show that:

- For the movie reviews experiment, high register texts are slightly more likely to be considered positive according to the model (CaCE = 0.00055).
- For the tweets experiment, high register texts are slightly less likely to be considered complaints according to the model (CaCE = -0.0054).

## Discussions and Possible Weaknesses

The project discusses several weaknesses and areas for improvement, including:

- The need for expert input in the text transformation process to ensure logical replacements.
- The limitations of computational resources, which prevented fully training the sentiment classification model.
- The possibility of considering multiple types of registers, such as formal/informal, slang, and vulgar.
- The importance of context-aware synonym replacements.
- The consideration that higher register text may inherently be less negative than lower register text, rather than being an unwanted bias.

## References

[1] A.Feder, N. Oved, U.Shalit, R.Reichart. CausaLM: Causal Model Explanation Through Counterfactual Language Models, arXiv:2005.13407v3, 2020.

[2] J.Pearl. CAUSALITY: MODELS, REASONING, AND INFERENCE, Cambridge University Press, 2000.

[3] G.Sohsah, M. Esad, G. Onur. Classification of Word Levels with Usage Frequency, Expert Opinions and Machine Learning, British Journal of Educational Technology, v46 n5 p1097-1101, 2015.

[4] Y.Zhu, R.Kiros, R.Zemel, R. Salakhutdinov, R.Urtasun, A.Torralba, ,Fidler. Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books, arXiv:1506.06724v1, 2015.
