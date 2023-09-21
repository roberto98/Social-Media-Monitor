# Social Media Monitor Tool
📚 "Natural Language Processing" project (Artificial Intelligence, UniGe)

## Overview

The goal of this project was to develop a social media monitoring tool to track topics and trends from various sources like news websites, blogs, and social media platforms. This enables businesses and individuals to stay informed about the latest developments and discussions related to their areas of interest. 

The project utilized various natural language processing (NLP) techniques to analyze unstructured textual data and extract insights. The main steps involved were:

- **Data Collection:** Gather data from various sources like news websites, blogs, and social media using APIs or web scraping techniques or RSS feed. In this case, the 20newsgroups dataset from Sklearn is used, which comprises around 18,000 newsgroups posts on 20 topics.

- **Text Preprocessing:** The text data was preprocessed to clean and normalize it. Techniques like removing stopwords, stemming, and lemmatization were used to prepare the text for further analysis. 

- **Topic Modeling:** Latent Dirichlet Allocation (LDA) was used to perform topic modeling. Both a custom implementation and the Gensim library's LDA were experimented with. LDA helped identify the main topics discussed in the corpus. Hyperparameter tuning was done to determine the optimal number of topics. The coherence score was used as the evaluation metric.

- **Sentiment Analysis:** The VADER sentiment analysis tool was used to determine positive, negative and neutral sentiment within the text data. This provided insights into the overall emotional tone of the content.

- **Summarization:** Extractive text summarization was implemented to create concise overviews of the most relevant content. It involved scoring sentences based on word frequencies and extracting the top scoring sentences.

- **Visualization and Reporting:** The results were visualized through charts, graphs and word clouds. These representations provided intuitive insights into topic distributions, sentiment analysis, and key themes. 


The project demonstrated how NLP techniques can be applied to analyze unstructured textual data from social media and extract useful insights around topics, trends, and sentiment. The techniques used provide a foundation to build more advanced social media analytics tools. 

A detailed report and a presentation on this project can be found here:
- [Social_Media_Monitor_Tool_report.pdf](https://github.com/roberto98/Social-Media-Monitor/files/12685046/Social_Media_Monitor_Tool_report.pdf)
- [Social_Media_Monitor_Tool_presentation.pdf](https://github.com/roberto98/Social-Media-Monitor/files/12685045/Social_Media_Monitor_Tool_presentation.pdf)



## How to Run
To run the Social_Media_Tool.ipynb file, follow these steps:

1. Download the Social_Media_Tool.ipynb file from the provided link.
2. Install the required libraries by running the following commands in your terminal or command prompt:
  ```
  pip install numpy
  pip install pandas
  pip install scikit-learn
  pip install nltk
  pip install gensim
  pip install pyLDAvis
  pip install vaderSentiment
  pip install wordcloud
  pip install matplotlib
  pip install tqdm
  ```
3. Open the Social_Media_Tool.ipynb file using Jupyter Notebook or any other compatible IDE.
4. Run the cells in the notebook sequentially, following the instructions and comments provided in the notebook.
