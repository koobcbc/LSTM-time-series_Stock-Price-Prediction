BERT-based Financial Sentiment Index and LSTM-based Stock Return Predictability

The BERT-based sentiment index outperforms traditional approaches (e.g., Multichannel CNN, BiLSTM, FastText) in precision, recall, and F1 score.

Demonstrates that LSTM (a nonlinear method) significantly outperforms traditional econometric models like Vector Autoregression (VAR).

Key Findings:

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


 Implications:

	1.	Investors and Analysts:
	•	Can use BERT-based sentiment indices for more accurate and real-time insights into market sentiment.
	•	Highlights the predictive power of integrating individual and institutional sentiment.
	2.	Model Developers:
	•	Demonstrates the superiority of advanced NLP (BERT) and deep learning (LSTM) methods for financial applications.
	3.	Future Research:
	•	Extending the analysis to broader markets.
	•	Addressing overfitting in longer-term stock return predictions.

 Conclusion:

The study establishes a novel framework by combining BERT and LSTM to build sentiment indices and predict stock returns. It highlights the superiority of these approaches over traditional econometric methods, emphasizing the nonlinear impact of investor sentiment on stock prices.
