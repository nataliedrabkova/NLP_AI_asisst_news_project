## IAEA News Filtering Pipeline: AI Assistant Prototype
### Project Objective
International organizations, such as the IAEA, often face the challenge of processing vast amounts of open-source information—particularly international news—to monitor global activities like nuclear trade and safeguards. Manual review of thousands of daily articles is highly time-consuming and difficult to scale.

### The Solution:
To address this challenge, this project uses the nuclear safeguards domain as a practical case study. It demonstrates a Proof of Concept for an "AI Assistant" designed to automate the ingestion of international press, extract key geopolitical entities (NER), translate text into semantic embeddings using Deep Learning, and classify each article's relevance to nuclear safety.

> ⚠️ Data Security Note: This project is an independent prototype developed strictly for demonstration purposes. It relies entirely on public, open-source news datasets and open-weight models. No actual, proprietary, or confidential data from the IAEA was accessed or used in this pipeline.

**[Dataset: all-the-news-2-1.csv](https://www.kaggle.com/datasets/davidmckinley/all-the-news-dataset/data?select=all-the-news-2-1.csv$0)** _(_source: Kaggle.com_)_

### Tech Stack & Methodologies
This pipeline was built to directly apply modern machine learning methodologies to real-world analytical challenges.
* **Large Language Models (LLMs) & Deep Learning:** Utilizing pre-trained transformer architectures to understand complex textual context without manual feature engineering.
* **Machine Learning Methodologies:** Implementing both Supervised Learning (Logistic Regression classification) and unsupervised concepts (Cosine Similarity search).
* **Libraries & Frameworks:**
  * `PyTorch` (Tensor operations and embedding generation)
  * `Hugging Face Transformers` (NER pipeline and Sentence-Transformers)
  * `Scikit-learn` (Classification and evaluation metrics)
  * `Pandas` & `Seaborn` (Data manipulation and visualization)

### Pipeline Architecture
The Jupyter Notebook is structured into a complete end-to-end analytical pipeline:
1. **Data Ingestion & Weak Supervision:** Loading a sample of 1,000 international news articles and applying programmatic keyword tagging (e.g., *nuclear, uranium, IAEA*) to generate initial training labels.
2. **Named Entity Recognition (NER):** Using a pre-trained BERT model to automatically extract relevant organizations (ORG), locations (LOC), and persons (PER).
3. **Semantic Text Embeddings:** Converting unstructured text into high-dimensional numerical vectors using PyTorch.
4. **Similarity Search:** Calculating mathematical similarity between documents to find semantically related news instantly.
5. **Supervised Classification:** Training a class-weight balanced Logistic Regression model on the text embeddings.
6. **Model Evaluation:** Visualizing performance through a Confusion Matrix to analyze the trade-off between False Positives and False Negatives in an imbalanced dataset.
7. **Real-time Inference:** Testing the model on brand-new, unseen data to simulate daily operations.

### Results & Performance
When tested on a subset of 1,000 international news articles, the classification model achieved an **overall accuracy of 84.5%**. 

More importantly, during live inference testing on newly simulated safeguards reports (e.g., *"International inspectors arrived in Vienna to discuss nuclear safety and uranium enrichment facilities"*), the model correctly flagged the text as highly relevant with a **99.90% confidence score**.

### How to Use
1. Clone this repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas torch transformers scikit-learn matplotlib seaborn
3. Download the All the News dataset from Kaggle and place the .csv file in your working directory (update the path in the notebook if necessary).
4. Run the Jupyter Notebook cell by cell to observe the data transformation, model training, and final AI Assistant filtering demonstration.

### Future Improvements
* **Scaling:** Executing the pipeline on the full dataset of 140,000+ articles using cloud compute.
* **Advanced Labeling:** Replacing weak keyword supervision with a Zero-Shot LLM classifier to create higher-quality ground-truth data for the supervised model.
