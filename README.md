# detecting-sexual-harassment
Built five different models to detect sexual harassment in text (primarily emails)

## Data Collection
### Harassment Data
1) Safe City (7,874)
2) Sexual harassment in academia (2,208); randomly sampled 300
3) Sexual harassment data from another study (99)
4) Sexual harassment in workplace from online sources compiled (huffpost, blogs) (107)

cleaned data by removing punctuation, numbers etc, removed data with length less than 5 and more than 400

Total: 7,978


### Non-harassment Data
1) New York Times (192,578)
2) Movie lines (9,885)

randomly sampled 20,000 data, cleaned data, and removed data with length less than 5 and more than 400

Total: 18,212


## Models
### Logistic Regression
#### 1) Bag of Words

	Used scikit-learn CountVectorizer to train, spaCy to preprocess data
	Accuracy: 98.9% on test data

#### 2) TF-IDF

	Used scikit-learn TfidfVectorizer to train, spaCy to preprocess data
	Accuracy: 98.6% on test data

### Recurrent Neural Network
	Used PyTorch to train LSTM and GRU models.
	Parsed 24,824 sentences.
	Found 44293 unique words tokens.
	Using vocabulary size 8000 + 1 (1 for padding)
	Batch size = 20
	Embedding dim = 400
	Hidden dim = 256
	N_layers = 1
	Lr = 0.001

##### Train: 80%, Validation: 10%, Test: 10%

#### 1) LSTM

		1. Embedding layer
		2. LSTM layer
		3. Fully connected layer
		4. Sigmoid activation layer
		5. Output

	Validation Accuracy: 98.0%
	Test Accuracy: 98.4%

#### 2) GRU

	1. Embedding layer
	2. Bidirectional GRU layer
	3. Fully connected layer
	4. Sigmoid activation layer
	5. Output

	Validation Accuracy: 0.836
	Test accuracy: 0.824

### Fasttext

	Used --pretrainedVectors wiki.en.vec
	epoch = 4
	lr = 0.4
	dim = 300
	Accuracy: 98.3%

