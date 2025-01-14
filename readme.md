# BERT-based Financial Sentiment Index and LSTM-based Stock Prediction

## Aim
To perform BERT-based sentiment index on news headlines from Google and Benzinga to predict future stock returns using LSTM-based non-linear model.

## Setup Instructions
```
$ pip install -r requirements.txt
$ python scripts/[name_of_file].py <input-dir> <output-dir>
```

## Methodology
1. News Headline and Stock Data Fetching & Integration
2. Data Cleaning
3. Sentiment Analysis
	- Sentiment polarity (positive, negative) is assigned to headlines using BERT, achieving high accuracy.
4. TFDistilBert Model Training
5. Forecasting & Evalution

## Research Paper that inspired me

1. Twitter mood predicts the stock market (https://www.sciencedirect.com/science/article/abs/pii/S187775031100007X)
2. Paper ()

## Dataset used

1. Sentiment Analysis Dataset from Kaggle
- This release of the financial phrase bank covers a collection of 4840 sentences. The selected collection of phrases was annotated by 16 people with adequate background knowledge on financial markets. Three of the annotators were researchers and the remaining 13 annotators were master’s students at Aalto University School of Business with majors primarily in finance, accounting, and economics.
2. Benzinga News API data
3. Yahoo Finance API data	

## Dataset

Sentiment Analysis Dataset (Train dataset)
- This release of the financial phrase bank covers a collection of 4840 sentences. The selected collection of phrases was annotated by 16 people with adequate background knowledge on financial markets. Three of the annotators were researchers and the remaining 13 annotators were master’s students at Aalto University School of Business with majors primarily in finance, accounting, and economics.

## Detailed Steps
1. Data Collection

a. Stock Price Data (NVDA):

	•	Use an API like Yahoo Finance to download NVIDIA stock price data.
	•	Collect historical data, including:
	•	Open, High, Low, Close prices (OHLC), Volume

b. News Sentiment Data:

	•	Use a financial news API like NewsAPI, Google News, or Twitter (for social media sentiment).
	•	Extract headlines, article summaries, or tweets related to NVIDIA.
	•	Apply BERT or FinBERT to classify the sentiment of each piece of text into categories like Positive, Neutral, or Negative.

2. Preprocessing the Data

a. Stock Price Features:

	•	Create lagged features (e.g., previous day’s closing price, moving averages).
	•	Normalize/standardize numerical data.

b. Sentiment Features:

	•	For each day:
	•	Aggregate news sentiment scores for that day (e.g., average sentiment).
	•	Use time alignment to match sentiment scores with stock price data.

3. Model Implementation

Models to Compare:

	1.	BERT + LSTM (Baseline from the Paper):
	•	Sentiment analysis using BERT on news data.
	•	Feed sentiment scores into an LSTM along with stock price features for prediction.
 In the future...
	2.	FinBERT:
	•	Use FinBERT for sentiment analysis instead of general-purpose BERT.
	•	Compare how its domain-specific training impacts prediction accuracy.
	3.	FinEAS:
	•	Fine-tune sentence embeddings with financial data and use them in a regression/prediction model.
	4.	FinLLaMA:
	•	Fine-tune LLaMA 2 for financial sentiment classification and include sentiment as features in the model.
	5.	Retrieval-Augmented LLMs:
	•	Incorporate external sources (e.g., real-time financial data) alongside sentiment to improve prediction robustness.

4. Evaluation Metrics

To compare models, consider:

	•	Prediction Accuracy: How close predictions are to actual stock prices.
	•	Mean Squared Error (MSE): Measures average squared difference between predicted and actual prices.

5. Workflow for Model Training

a. Feature Engineering:

	•	Combine numerical stock price data with sentiment scores.

b. Training and Testing:

	•	Split data into train, validation, and test sets (e.g., 70%-20%-10%).
	•	Fine-tune each model on the training data and validate it on the test set.

c. Implementing Models:

	•	BERT + LSTM:
	•	Train a BERT model for sentiment classification.
	•	Feed sentiment features and stock data into an LSTM for price prediction.

6. Tools and Libraries

	•	Data Collection:
	•	yfinance, pandas_datareader, or APIs for stock prices.
	•	NewsAPI, tweepy for news and social media data.
	•	Machine Learning Frameworks:
	•	transformers (for BERT, FinBERT, and LLaMA).
	•	TensorFlow or PyTorch (for LSTM).
	•	Scikit-learn (for evaluation metrics).
	•	Deep Learning Libraries:
	•	Hugging Face Transformers for pre-trained NLP models.
	•	keras or pytorch-lightning for LSTM implementation.

## Key Findings:

	1.	Sentiment Extraction:
	•	Sentiment polarity (positive, neutral, negative) is assigned to Weibo posts using BERT, achieving high accuracy.
	•	Textual sentiment is linked to individual investors, while option and market sentiment represent institutional and combined influences, respectively.
	2.	Model Performance:
	•	BERT outperformed other machine learning (ML) and traditional lexicon-based methods.
	•	LSTM captured nonlinear relationships and demonstrated better stock return prediction performance than VAR.
	3.	Stock Return Predictions:
	•	Combining all three sentiment indices improves prediction accuracy.
	•	Sentiment indices enhance predictions even when controlling for traditional risk factors (e.g., Fama-French factors).
	4.	Limitations:
	•	Overfitting in LSTM models observed for longer-term predictions.
	•	Findings are restricted to individual stock-level analysis, and market-level generalization requires future research.

## Helpful Resources:

## Implications:

	1.	Investors and Analysts:
	•	Can use BERT-based sentiment indices for more accurate and real-time insights into market sentiment.
	•	Highlights the predictive power of integrating individual and institutional sentiment.
	2.	Model Developers:
	•	Demonstrates the superiority of advanced NLP (BERT) and deep learning (LSTM) methods for financial/other applications.
	3.	Future Research:
	•	Extending the analysis to broader markets.
	•	Addressing overfitting in longer-term stock return predictions.
 
 ## Conclusion:

The study establishes a novel framework by combining BERT and LSTM to build sentiment indices and predict stock returns. It highlights the superiority of these approaches over traditional econometric methods, emphasizing the nonlinear impact of investor sentiment on stock prices.

## Future Steps:

Some other novel models to compare:

1. FinBERT: Financial Sentiment Analysis with BERT

•	Release Date: August 2019
•	Reference: ArXiv, 1908.10063

FinBERT is a domain-specific adaptation of BERT, pretrained on a large financial corpus to better understand financial terminology and context. It has demonstrated superior performance in financial sentiment analysis tasks compared to the original BERT model. (Ar5iv)
[https://arxiv.org/abs/1908.10063](https://arxiv.org/pdf/1908.10063)



2. FinLlama: Financial Sentiment Classification for Algorithmic Trading

	•	Release Date: July 2023
	•	Description: A fine-tuned version of the LLaMA 2 (7B) model focused on financial sentiment classification, leveraging the generative capabilities of LLaMA.
	•	Reference: ArXiv, 2403.12285
	•	Developed By:
	•	Independent researchers and contributors leveraging the LLaMA 2 architecture by Meta AI.
	•	Specific details about the FinLLaMA fine-tuning contributors have not been highlighted publicly yet.
	•	Institution: Likely independent or part of a broader research collaboration utilizing open-source LLaMA.
	•	Reference: FinLLaMA on ArXiv

[https://arxiv.org/abs/2111.00526](https://arxiv.org/pdf/2111.00526)

3. Retrieval-Augmented Large Language Models


	•	Key Paper: October 2023
	•	Description: Integrates large instruction-tuned language models with retrieval systems to fetch external context for improved sentiment prediction. Retrieval-augmented LLMs show better generalization and performance, especially in knowledge-intensive tasks.
	•	Reference: ArXiv, 2310.04027
	•	Developed By:
	•	Researchers at Google Research and OpenAI contributed to the general field of retrieval-augmented LLMs.
	•	Specific implementations for financial sentiment may come from private firms or academic labs fine-tuning their own models.
	•	Institution: General methods often stem from Google Research, OpenAI, and others; specific applications may vary.
	•	Reference: ArXiv, 2310.04027

Recent studies have explored enhancing financial sentiment analysis by integrating retrieval-augmented LLMs. This methodology combines instruction-tuned LLMs with retrieval mechanisms that fetch additional context from external sources, leading to significant improvements in accuracy and F1 scores. (arXiv)

[https://arxiv.org/abs/2310.04027](https://arxiv.org/pdf/2310.04027)

4. FinEAS: Financial Embedding Analysis of Sentiment

	•	Release Date: November 2021
	•	Description: Focuses on sentence embeddings tailored for financial text sentiment analysis, achieving improvements over traditional BERT and FinBERT models in financial classification tasks.
	•	Reference: ArXiv, 2111.00526
	•	Developed By:
	•	Researchers at the University of Massachusetts Amherst and Fidelity Investments.
	•	Lead contributors: Yi Yang, Mark Christopher Yates, and Preslav Nakov.
	•	Institution: Collaboration between academia (UMass Amherst) and industry (Fidelity Investments).
	•	Reference: FinBERT on ArXiv

FinEAS introduces a new language representation model tailored for financial sentiment analysis. By fine-tuning sentence embeddings from a standard BERT model, FinEAS achieves notable improvements over vanilla BERT, LSTM, and even FinBERT in sentiment classification tasks. (arXiv)


https://arxiv.org/abs/2111.00526

These advancements indicate a trend towards leveraging specialized LLMs and retrieval-augmented techniques to enhance the accuracy and efficiency of financial sentiment analysis, surpassing the performance of earlier BERT+LSTM models.
