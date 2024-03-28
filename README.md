# Propaganda-Detection-System

"Propaganda Detection System" using NLP techniques on a 358 MB dataset with 166,000 entries and 10 features. Applied extensive EDA for data visualisation and utilised data cleaning tools for batch processing, unification, scaling, formatting, encoding, and text array augmentation. Implemented lemmatisation and stemming for input to simple classification models, and various ANN architectures; and explored BERT models with different pre-trained embeddings.


## Problem definition and methodology

Propaganda is currently a significant concern, especially in the era of widespread social media usage. The more connected people become through these platforms, the more susceptible they are to various forms of propaganda. Within this project, an NLP model has been developed to effectively detect any instances of propaganda within given texts or prompts. This particular task falls under the supervised classification problem category. By popularising the issue and utilising effective problem-solving methods, substantial assistance can be gradually provided to individuals worldwide in order to prevent the spread of misinformation. This study focuses on various works regarding the detection of Propaganda and modern applications of NLP in such projects. It also compares different classification models and provides an overview of the implementation of BERT to address this problem.

Problem examples:

- L. Glendinning, "Exposing Cambridge Analytica: 'It's been exhausting, exhilarating, and slightly terrifying," The Guardian, 2018.
  
- E. Briant,"Three Explanatory Essays Giving Context and Analysis to Submitted Evidence of the Cambridge Analytica 's involvement in the US elections 2014," Propaganda and Counter-terrorism: Strategies for Global Change, Manchester: Manchester University Press, 2015.
  
- K. Hill, "YouTube Science Scam Crisis," 26 May 2023.  [Online]. Available from: https://www.youtube.com/watch?v=McM3CfDjGs0. [Accessed August 2023].

Supervised learning is considered for this project because there are labeled datasets available, and it seems hard to find a proper way to implement unsupervised learning for this task. For our case, I have opted to work with labelled data from the beginning. Nevertheless, it would be an intriguing perspective to replicate the same experiment without any labels. For the equipment, Google Colab was used along with a Colab Pro+ subscription and additional compute units, as certain models and calculations took more than a day to run with the free version. For data storage, a Google Drive Basic (100 GB) subscription was acquired due to the limited 15 GB memory in the free version. The machine's specifications were as follows: System - Linux 64-bit, Programming Language - Python 3.10.12.


## Dataset

Four datasets related to misinformation were chosen from Kaggle, fact-checked internet posts, fake news, and propaganda in general. It is more convenient to do so rather than manually collecting and annotating all the necessary information. The main dataset is a combination of the following datasets:

- steven, "Misinformation & Fake News text dataset 79k," Kaggle, 2022. [Online]. Available from: https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k. [Accessed August 2023].
  The dataset "Misinformation & Fake News text dataset 79k" (233.07 MB) consists of 85,957 rows. Among them, 34,975 are considered 'true' and originate from sources such as Reuters, the New York Times, the Washington Post, and others. The remaining 51,011 entries are classified as 'fake' and are sourced from American right-wing extremist websites including Redflag Newsdesk, Beitbart, Truth Broadcast Network. Additionally, misinformation and propaganda data were collected by the EUvsDisinfo project, a website dedicated to fact-checking information. Another source of fake news is a public dataset from the article titled "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques" by Ahmed, Hadeer, Traore, Issa, and Saad (2017). This article presents a fake news detection model that utilises n-gram analysis, six different machine learning techniques, TF-IDF, and LSVM. The dataset contains three Kaggle .CSV files, two of which contain propaganda and one that contains factual information. Each file consists of two attributes, namely, index and text.

- Matti Ur Rehman, "Verified Posts: Fact-Checking Online Content," Kaggle, March 2023. [Online]. Available from: https://www.kaggle.com/datasets/mattimansha/verified-posts-fact-checking-online-content. [Accessed August 2023].
  The second dataset, "Verified Posts: Fact-Checking Online Content" (5.91 MB), comprises posts from social media platforms and covers the period from 2008 to 2022. This dataset is represented by a single .CSV file, containing 22,020 rows and 5 columns: object (e.g., Facebook post), date, web-link, text, and Truth-O-Meter rating which indicates the accuracy of the statement. The data originates from the PolitiFact website, another reputable source for fact-checking information. The website employs Truth-O-Meter and Flip-O-Meter ratings to describe the consistency and truthfulness of information. Truth-O-Meter encompasses 6 categories: True (the statement is accurate and does not have any significant omissions), Mostly True (the statement is accurate but requires clarification or additional information), Half True (the statement is partially accurate but excludes important details or presents information out of context), Mostly False (the statement contains some truth but disregards critical facts leading to a different impression), False (the statement is inaccurate), and Pants On Fire (the statement is not accurate and makes an absurd claim). Flip-O-Meter assesses an official's consistency on a given issue and comprises 3 categories: No Flip (no significant change in position), Half Flip (a partial change in position), and Full Flop (a complete change in position).
  
- Clément Bisaillon, "Fake and real news dataset," Kaggle, 2020. [Online]. Available from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset. [Accessed August 2023].
  The third dataset, called "Fake and real news dataset," is 116.37 MB in size and contains two .CSV files. One file contains fake information, while the other consists of true information. The dataset primarily focuses on news
articles, with over half of them being about US politics, while the rest cover global news. In total, there are 44,898 entries in the dataset.
  
- Bohdan Mynzar, yoctoman, "Twitter Ru Propaganda Classification ," Kaggle, May 2023. [Online]. Available from: https://www.kaggle.com/datasets/bohdanmynzar/twitter-propaganda-classification. [Accessed August 2023].
  The last dataset, called "Twitter Ru Propaganda Classification," is 2.43 MB in size and contains only one file. Although there is another file available, it is in a language other than English, so only the first file was used. This dataset comprises Twitter posts from various sources in the year 2022. The sources of the 'fake' information are RT (Russia Today) Twitter account, the Russian Ministry of Foreign Affairs official Twitter account, and other similar sources. On the other hand, the sources of the 'true' information are considered to be BBC World, Bloomberg Politics, The Kyiv Independent, and others. This dataset was collected and provided by the Faculty of Informatics and Computer Engineering at the National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute" in Kyiv, Ukraine. The data was used in the publication "Machine Learning Method for Detecting Propaganda in Twitter Texts" (2023) authored by B. Mynzar, I. Stetsenko, Y. Gordienko, S. Stirenko. The dataset contains 12,990 rows and 5 columns, which include index, id, date, text, and a true/false criterion.

- !!! It is important to acknowledge that all the aforementioned Kaggle datasets intrinsically contain bias, particularly when the data is not self-defined. The assessment of intricate definitions and borderline subjects ultimately lies with the individual researcher. Nevertheless, it is worth noting that any dataset, even if self-constructed, will inherently exhibit a certain level of bias. What truly matters is the degree to which objectivity is twisted by a bias. Consequently, models can become highly inaccurate or accurate for erroneous reasons. Hence, it is crucial to mitigate bias and enhance model generalization. Potential biases can arise from the sources or intentions of dataset creators in this study; however, these biases appear to fall within the confines of the normal margin of error. Another crucial aspect is that although the dataset is sufficiently large, it
remains constrained by its sources of origin and shaped by the overarching themes surrounding the primary subjects of these datasets. Nonetheless, a substantial amount of compiled data exists in terms of quantity and
characteristics. The potential of these datasets will be further explored in the following paragraph.


## Analysis

All datasets were standardised to a common format consisting of two columns: a "text" column containing the actual text and a "propaganda" column indicating whether the text is propagandistic. The values in the "propaganda" column are 1 for propaganda and 0 for non-propaganda. After combining all four datasets, one additional row has been added and basic EDA has been performed. As a result, almost all entries have less than 2,500 words. And the most frequent words include: Donald Trump, US, one, people, will, say... This demonstrates that a significant portion of the data pertains to the United States. The largest entry comprises 24,234 words, with an average length of approximately 343 words per row.

The next essential step is to clean the dataset. There are 12 main functions used for this purpose:

- Batching Function (In this case, the batch size is set to 10,000 - this happens to be the good equivalent.)
  
- Special Character Deletion (To accomplish this, the re library was imported.)
  
- Text Encoding (ASCII format.)
  
- Stop-Word Removal (To facilitate the removal of stop-words, prebuilt natural language processing (NLP) libraries and tools frequently consist of lists of stop-words in different languages, such as NLTK in this project.)
  
- Lowercase Conversion
  
- White Space and NaN Value Removal
  
- Text Segmentation Function (Dividing the text into equal pieces while ensuring that each piece does not surpass 512 characters is necessary for the final model, BERT, as it only accepts text within this length limit.)
  
- 'Augmentation' Function (Numerous augmentation methods are available, including synonym replacement, back-translation, perturbation using contextual word embeddings, and more. For the project, a new dataset is created by
segmenting the text into distinct rows, with the same index assigned to indicate their common text origin.)

- Scaling Function (Common scaling methods in machine learning include Min-Max Scaling, Standardisation (Z-score Scaling), and Robust Scaling. The choice of scaling method depends on the data characteristics and algorithm requirements. Scaling ensures effective performance and fair feature comparisons. Scaling the index parameters by dividing them by the maximum index number works well for training simple models. StandardScaler is suitable for Gaussian distributed data, while Min-Max Scaler is useful for scaling data to a specific range. Scaling by the highest value yields optimal results for Machine Learning, even though it is sensitive to outliers, which are absent in this case. In this project, the technique is used to scale the text indexes that demonstrate the relation of text chunks across various rows. A more thorough discussion on this matter will follow.)
  
- Lemmatisation (Various lemmatisation libraries are available, but they can be memory-intensive. Despite this, employing lemmatisers has shown significant improvements in validation scores. NLTK, with its WordNet interface, provides access to lexical databases and resources for word sense disambiguation and synonym resolution. WordNet is invaluable for language resource construction. NLTK WordNet is frequently employed for semantic analysis,
sentiment analysis, and information retrieval tasks that place significance on comprehending word relationships and meanings. This work provides an overview of different NLP tools and their functionalities. It mentions the use of WordNet and POS tagging for identifying word relationships based on their parts of speech. It also introduces spaCy as a comprehensive NLP library that offers various text processing capabilities. TextBlob is highlighted as a user-friendly NLP library with straightforward APIs for common tasks like tokenisation, POS tagging, and sentiment analysis. Lastly, RNNTagger, a type of POS tagger that uses recurrent neural networks, is mentioned for automatic POS tagging and its advantages in syntactic analysis, text classification, and named entity recognition.)

- Stemming (Various stemming algorithms, including the Porter Stemmer, Snowball Stemmer, and Regexp Stemmer, provide varying degrees of accuracy and customisation. For this classification task, the Porter Stemmer was selected due to its basic nature.)
  
- Tokenisation and Vectorisation (Common text vectorisation techniques include BoW, TF-IDF, Word Embeddings, and Document Embeddings. Different techniques have different strengths and limitations. Experimentation with six transformers and tokenisers: TfidfVectorizer, TfidfTransformer, Keras Tokenizer, RegexpTokenizer, Word2Vec, and Doc2Vec were conducted in this project. These were used for various purposes such as TF-IDF feature extraction, converting text into numerical sequences, tokenising text using regular expressions, generating word embeddings, and generating document or sentence embeddings. Some models have pre-trained datasets, while others undergo training on provided datasets. It is important to allocate sufficient memory resources for these models. Excessive memory usage may lead to session termination.)


## Models

For the baseline Classification models, a selection of algorithms has been made: Naive Bayes, SGD Classifier, SVM, Random Forest, MLP Classifier, and Single Layer NN. Each algorithm brings its unique strengths to the project:

- Naive Bayes is a simple probabilistic algorithm that is particularly well-suited for text supervised classification tasks. It excels at handling text data due to its inherent assumptions about independence.
  
- Whereas, SGD Classifier is an efficient classifier that performs admirably on large datasets and in online learning scenarios. It's a great choice when dealing with vast amounts of data and when real-time adjustments are necessary.
  
- SVM, on the other hand, is a powerful model that excels in separating classes with optimal hyperplanes. It's particularly useful when dealing with complex decision boundaries.

- Random Forest, as an ensemble method, is effective in providing robust classification. By combining multiple decision trees, it can handle noisy data and avoid overfitting.

- MLP Classifier can model non-linear relationships and interactions between features very effectively. It is flexible in architecture and allows customisation to match the complexity of the problem.
  
- Lastly, Single Layer NN is a simple linear model that is particularly suitable for linearly separable data. While it may not handle intricacies as well as other larger models, it excels in straightforward scenarios.

The highest scores among the base models were achieved by Random Forest (Training Accuracy: 100%, Validation Accuracy: 58.66%), MLP Classifier (Training Accuracy: 99.98%, Validation Accuracy: 58.23%), and Single Layer NN (Training Accuracy: 87.93%, Validation Accuracy: 60.90%). The Random Forest and MLP Classifier models and their performance scores based on Tfidf Vectoriser combined with Data Cleaning, and different lemmitisors or stemming. Despite negligible differences in performance results, this indicates that the main differentiating factor lies in the architecture of the models and other tuning variations. Furthermore, the training results of the Single Layer NN model which does not exhibit the highest scores for Training Accuracy, but demonstrates the greatest Validation Accuracy among all the baseline models. An evident observation is the overfitting present in the model. Slight improvements can be expected through better tuning; however, a more profound enhancement in performance requires the inclusion of additional layers. 

In this research, various embeddings were employed, including:

- small_bert/bert_en_uncased_L-4_H-512_A-8 (Is a BERT-based model commonly used for NLP tasks.)
  
- glove.840B.300d (Refers to word embeddings trained using the GloVe algorithm.)

- GoogleNews-vectors-negative300 (Denotes word embeddings developed by Google through Word2Vec.)

- wiki-news-300d-1M-subword (Pertains to FastText word embeddings trained on Wikipedia text, incorporating subword information.)

For more advanced models, RNN, LSTM, Bidirectional LSTM, and BERT have been selected:

- RNN sequentially processes sequences and faces challenges in handling long-range dependencies.
  
- On the other hand, LSTM overcomes the vanishing gradient problem and effectively captures long-range dependencies.

- Bidirectional LSTM processes sequences in both directions, allowing it to capture bidirectional context.

- BERT works on a transformer architecture to capture contextualised word representations and has demonstrated outstanding performance across various NLP tasks.

That is thereon for BERT to be elected as the main model for the project. BERT takes into account the context of each word in a sentence from both directions, in contrast to traditional models that process text unidirectionally. This bidirectional method enables BERT to more effectively capture context. BERT adopts an encoder-based architecture that transforms input tokens into dense vector representations, effectively capturing the meaning of the text. It employs self-attention mechanisms, based on the transformer architecture, to determine word importance by examining their relationships within a sentence. This enables BERT to capture long-range dependencies and context. During pre-training, BERT learns to predict missing words (Masked Language Model task) and understand sentence relationships (Next Sentence Prediction task) using extensive text data. Consequently, BERT produces contextualised word embeddings that represent language characteristics. BERT's ability to generate contextually relevant representations has had a huge impact on NLP, excelling in various tasks including text classification, named entity recognition, question answering, sentiment analysis, and language translation. When fine-tuning BERT is used for specific tasks, less time and data are required compared to training separate models.

Several issues arose during the realisation of architects of different Bert model, mainly due to incompatibility between program versions. As a result, numerous implementations of the BERT model, which also have not received updates in recent years, do not function with Keras or TensorFlow libraries. Google Colab possesses limited repository of diverse library versions but remains incomplete. Occasionally, the situation may demand the installation of a newer library version into Google Colab, as is the case with this project. After updated versions of the TensorFlow library were employed, text cleaning by Re library was implemented to remove non-standard characters. The vocabulary and embedding dimensions were set to 10,000, moderating the model's learning time. In order to divide the text into smaller portions, the strings were limited to 70 words with a maximum of 80 words, with the last 10 words of each subset becoming the beginning of the next one. Augmentation was then performed, resulting in the creation of a new dataset with three additional columns: index, text, and propaganda. The index column was introduced to assign identical values to rows containing text from the same original text. Consequently, the length of the new dataset is nearly double that of the original one. This expanded dataset was divided equally into two halves - training data and test data - with various approaches, such as a 30% vs 70% or 20% vs 80% split, being considered. This partitioning plays a crucial role in model training. Having more training data results in a more precise model closely related to the specific dataset. For the embedding dictionary, the "small_bert/bert_en_uncased_L-4_H-512_A-8" dataset was employed. This dataset, which is widely used in various works, offers a great balance between model size and capability. Larger Transformers require additional space and computational power, both of which are limited resources. However, these heavier models are known to improve the overall accuracy. The aforementioned embedding is trained using the Wikipedia and BooksCorpus English datasets (103.18MB). 

The classifier model's text input layer is constructed next, followed by the preprocessing Keras second layer, the BERT encoder Keras third layer, the dropout fourth layer, and finally the Dense classifier fifth and final layer. The dropout value is set to 0.1 without any activation. However, it is commonly recommended to use Sigmoid activation and Adam optimiser for binary classification problems. The metric for calculating loss is set as Binary Accuracy and Binary Cross-entropy. Afterwards, the main dataset is partitioned into four nearly equal-sized folders based on volume, sorted in GoogleDrive, with each text string assigned to an individual folder. There are four folders for test propaganda and non-propaganda, as well as train propaganda and non-propaganda texts. These files will be extracted from the folders with PrefetchDataset - function of TensorFlow library that creates datasets that asynchronously prefetch elements from 'input_dataset'. The next step involves batching the data and splitting the training dataset into training and validation sets with a proportion of 20 to 80, where 20% of the dataset is allocated to the validation set. The random seed used in this workbook is always 42. With the help of Keras functions, four prefetch objects are created to extract the data. For training the model with this specific dataset, it is recommended to use 5 Epochs, which is sufficient to prevent the training from running for more than 2 hours while producing good results. However, towards the end of the training, the model did not exhibit significant improvement with these parameters. The default learning rate for the optimiser is 3e-5. Depending on the optimiser being used, this value could also be 3e-4, 1e-4, 0, or other values, but they did not yield much improvement in the model's accuracy in this particular situation. Extracting the cardinality of the datasets is necessary for determining the steps involved in the learning process for each Epoch. The optimiser is created using the AdamW stochastic optimisation method. Finally, the supervised classification model is compiled, and the fitting and training process is proceeded with. After many hours and evaluating the model on the evaluation dataset following the training, the best results thus far are achieved, with an accuracy of 83.72%. The results indicate that the model is exhibiting overfitting, particularly towards the later stages of the training process. More data might contribute to its improvement.


## 6. Conclusion

A small yet sufficiently robust model was developed, showing optimal outcomes during training. The primary objectives of comprehending and mastering its functionality were accomplished. Although there is room for improvement, particularly in terms of accuracy. Improving fine-tuning techniques and employing targeted data augmentation and processing methodologies can contribute to a marginal increase in overall accuracy. To greatly enhance the training, it is necessary to use an alternative model. In the realm of NLP, BERT is already considered an outdated model, serving as a benchmark for more sophisticated models such as ALBERT, DeBERTa, FinBERT, RoBERTa, DistilBERT, and even non-BERT architectural models including T5 (Text-to-Text Transfer Transformer), XLNet, ELECTRA, UniLM, and Megatron. ALBERT, also known as A Lite BERT, tackles the limitations of BERT through the implementation of parameter reduction techniques. This approach aims to maintain or even enhance performance. On the other hand, DeBERTa broadens BERT's capabilities by seamlessly integrating both local and global context within a unified model. As a result, DeBERTa exhibits improved performance across different tasks. FinBERT is a variant of BERT that focuses on sentiment analysis within the financial domain. It exhibits an understanding of specialised investment terminology and effectively determines sentiment. The training data of FinBERT integrates text from financial news services and the FiQA dataset. 

As for RoBERTa, it refers to an acronym denoting the Robustly optimised BERT approach. Essentially, RoBERTa shares the same architecture as BERT, but it introduces additional improvements in terms of resources and time. These enhancements arise primarily from an enhanced training methodology, augmented computational power, and significantly larger training data, amounting to an enormous increase. Additionally, RoBERTa includes dynamic masking, which adapts throughout the training process. DistilBERT, on the other hand, represents a condensed version of BERT that relies on half the number of training examples needed for BERT, while still maintaining almost 100% of the overall performance. This approach employs a distillation technique to approximate the original BERT within a significantly smaller model. Notably, this approach offers a more sustainable training process, yet still demanding substantial computational resources. To utilise such superior models, it is imperative to possess more potent equipment - computational power capable of accommodating days of training duration. Another intriguing aspect involves engaging in unsupervised ML to mitigate the impact of human biases as extensively as possible. As technology and techniques of propaganda continue to evolve, ongoing researches are dedicated to countering emerging methods of manipulation and deception. The history of AI-driven propaganda detection evolves in parallel with technological advancements and the changing landscape of online information. Research, innovation, and collaboration remain crucial in the persistent fight against misinformation and propaganda.


## Inspired by

Numerous sources have served as inspiration, and I am grateful to those who assisted, motivated, and supported the creation of this work. Several interesting and relevant projects are available on the Interent, some of
which have contributed to the dataset used in this project. This work has gained valuable insights from these projects:

- Jacob Devlin and Ming-Wei Chang, "Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing," Google AI Language, 2 November 2018. [Online]. Available from: https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html. [Accessed August 2023].
  
- Bohdan Mynzar (Owner) and yoctoman (Viewer), "Propaganda Binary Classification," Kaggle, 25 June 2023. [Online]. Available from: https://www.kaggle.com/code/bohdanmynzar/propaganda-binary-classification/notebook. [Accessed August 2023].
  
- Tetsuya Sasaki, "How can we classify FAKE news?" Kaggle, 2 June 2022. [Online]. Available from: https://www.kaggle.com/code/sasakitetsuya/how-can-we-classify-fake-news/notebook. [Accessed August 2023].
  
- Atish Adhikari, "Fake-News Cleaning+Word2Vec+LSTM (99% Accuracy)," Kaggle, 25 April 2020. [Online]. Available from: https://www.kaggle.com/code/atishadhikari/fake-news-cleaning-word2vec-lstm-99-accuracy. [Accessed August 2023].
  
- Husein Zolkepli, "Type and Magazine Text Classification," Kaggle, 3 September 2017. [Online]. Available from: https://www.kaggle.com/code/huseinzol05/type-and-magazine-text-classification. [Accessed August 2023].
  
- Husein Zolkepli, "Type and Magazine Text Classification," Kaggle, 3 September 2017. [Online]. Available from: https://www.kaggle.com/code/huseinzol05/type-and-magazine-text-classification. [Accessed August 2023].
  
- ibrahim, "Text Classification on ISIS Quote," Kaggle, 17 January 2019. [Online]. Available from: https://www.kaggle.com/code/amnibrahim/text-classification-on-isis-quote. [Accessed August 2023].
  
- bentrevett Ben Trevett, sejas Antonio Sejas, "PyTorch PoS Tagging," 4 Jun 2021. [Online]. Available from: https://github.com/bentrevett/pytorch-pos-tagging. [Accessed August 2023].
  
- CoreNLP. Stanford NLP Group. 2020. [Online]. Available from: https://stanfordnlp.github.io/CoreNLP/index.html. [Accessed August 2023].
  
- jacobdevlin-google, "TensorFlow code and pre-trained models for BERT," 11 March 2020. [Online]. Available from: https://github.com/google-research/bert. [Accessed August 2023].
  
- Harsh Jain, "BERT for "Everyone" (Tutorial + Implementation)," 20 May 2021. [Online]. Available from: https://www.kaggle.com/code/harshjain123/bert-for-everyone-tutorial-implementation/notebook. [Accessed August 2023].
  
- Detect_fake_news.ipynb. [Online]. Available from: https://colab.research.google.com/github/singularity014/BERT_FakeNews_Detection_Challenge/blob/master/Detect_fake_news.ipynb#scrollTo=QBGl2JwxsFxs. [Accessed August 2023].
  https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/classify_text_with_bert.ipynb#scrollTo=Cb4espuLKJiA. [Accessed August 2023].

- Training a part-of-speech tagger with transformers (BERT). [Online]. Available from: https://colab.research.google.com/github/explosion/thinc/blob/master/examples/02_transformers_tagger_bert.ipynb#scrollTo=y9KODZffe5hK. [Accessed August 2023].
