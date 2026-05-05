from __future__ import annotations

import re


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


SAMPLE_ARTICLES: list[dict[str, str]] = [
    {
        "title": "Artificial intelligence",
        "text": (
            "Artificial intelligence is the field of computer science that studies systems able to perform tasks "
            "that normally require human intelligence. These tasks include reasoning, learning, planning, perception, "
            "language understanding, and decision making. Modern artificial intelligence combines algorithms, data, "
            "statistical models, and computing power to solve problems.\n\n"
            "Machine learning is a major subfield of artificial intelligence. Instead of being explicitly programmed "
            "for every rule, a machine learning system learns patterns from examples. Deep learning uses neural "
            "networks with many layers and has become important for computer vision, speech recognition, natural "
            "language processing, and generative AI.\n\n"
            "Artificial intelligence systems are evaluated by their accuracy, robustness, fairness, interpretability, "
            "and ability to generalize to new situations. Practical AI applications include search engines, medical "
            "decision support, recommendation systems, fraud detection, autonomous vehicles, and conversational agents."
        ),
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    },
    {
        "title": "Machine learning",
        "text": (
            "Machine learning is a branch of artificial intelligence concerned with algorithms that improve through "
            "experience and data. A learning algorithm builds a model from training examples and uses the model to "
            "make predictions or decisions on new data. Common tasks include classification, regression, clustering, "
            "ranking, anomaly detection, and recommendation.\n\n"
            "Supervised learning uses labeled examples, such as emails marked as spam or not spam. Unsupervised "
            "learning looks for structure in unlabeled data, such as groups of similar customers. Reinforcement "
            "learning trains an agent to choose actions by receiving rewards or penalties from an environment.\n\n"
            "A good machine learning workflow includes data preparation, feature engineering or representation "
            "learning, model training, validation, testing, and monitoring after deployment. Overfitting happens when "
            "a model memorizes training data too closely and performs poorly on unseen data. Cross-validation, "
            "regularization, simpler models, and more data can reduce overfitting."
        ),
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
    },
    {
        "title": "Artificial neural network",
        "text": (
            "An artificial neural network is a machine learning model inspired by networks of biological neurons. "
            "It contains connected units called neurons or nodes. Each connection has a weight, and training adjusts "
            "the weights so that the network produces useful outputs from input data.\n\n"
            "Neural networks are organized in layers. A feedforward network passes information from input layers to "
            "hidden layers and finally to an output layer. Deep neural networks contain multiple hidden layers, which "
            "allow them to learn complex representations. Backpropagation calculates how much each weight contributed "
            "to an error, and gradient descent updates the weights to reduce that error.\n\n"
            "Convolutional neural networks are commonly used for images because they can detect local visual patterns. "
            "Recurrent neural networks and transformer architectures are used for sequences such as text, audio, and "
            "time series. Neural networks can be powerful, but they often require large datasets and careful validation."
        ),
        "url": "https://en.wikipedia.org/wiki/Artificial_neural_network",
    },
    {
        "title": "Information retrieval",
        "text": (
            "Information retrieval is the process of finding relevant information from a collection of documents. "
            "Search engines, library catalogs, and question answering systems all rely on information retrieval. "
            "The user submits a query, and the system ranks documents or passages that are likely to answer the query.\n\n"
            "Classical retrieval methods include inverted indexes and term-based scoring such as BM25. Neural retrieval "
            "uses embeddings, which represent text as vectors. Vector search can retrieve passages with similar meaning "
            "even when they do not share the exact same words. Hybrid search combines lexical matching and vector "
            "similarity to improve recall and precision.\n\n"
            "Retrieval augmented generation, or RAG, connects an information retrieval system to a language model. "
            "The retriever selects relevant sources, and the generator writes an answer grounded in those sources. "
            "A self-corrective RAG pipeline can judge whether retrieved sources are useful, reformulate the query, "
            "and verify that the final answer is supported by the evidence."
        ),
        "url": "https://en.wikipedia.org/wiki/Information_retrieval",
    },
    {
        "title": "Overfitting",
        "text": (
            "Overfitting is a machine learning problem where a model learns the training data too closely, including "
            "noise, accidental patterns, and examples that do not generalize. An overfit model can have very high "
            "training accuracy but poor performance on new data. This happens when the model is too complex for the "
            "amount or quality of data available.\n\n"
            "Overfitting is detected by comparing training performance with validation or test performance. If the "
            "training error is low but validation error is high, the model is probably overfitting. Cross-validation "
            "is often used to estimate how well a model will generalize to unseen examples.\n\n"
            "Common ways to reduce overfitting include using more training data, simplifying the model, applying "
            "regularization, early stopping, pruning decision trees, dropout in neural networks, and data augmentation. "
            "The opposite problem is underfitting, where the model is too simple and cannot capture the important "
            "patterns in either the training data or the test data."
        ),
        "url": "https://en.wikipedia.org/wiki/Overfitting",
    },
    {
        "title": "BM25",
        "text": (
            "BM25 is a ranking function used by search engines and information retrieval systems to estimate how "
            "relevant a document is to a query. It is based on lexical matching, which means it scores documents by "
            "looking at the query terms that appear in the document text. BM25 is a strong classical baseline for "
            "retrieval augmented generation systems.\n\n"
            "BM25 improves on simple term frequency scoring by using inverse document frequency and length "
            "normalization. Rare query terms usually receive more weight than common terms, and very long documents "
            "are controlled so they do not win only because they contain many words. This makes BM25 useful for "
            "retrieving exact terminology such as model names, algorithms, concepts, and technical definitions.\n\n"
            "Hybrid search often combines BM25 with vector search. BM25 catches exact words and names, while vector "
            "search catches semantic similarity. In a RAG pipeline, BM25 can retrieve passages that mention the query "
            "directly, and reranking can then reorder those passages before answer generation."
        ),
        "url": "https://en.wikipedia.org/wiki/Okapi_BM25",
    },
    {
        "title": "Vector search",
        "text": (
            "Vector search is a retrieval technique that represents text, images, audio, or other objects as numeric "
            "vectors called embeddings. Similar objects are placed close together in the vector space. A query is also "
            "embedded as a vector, and nearest neighbor search finds documents whose vectors are closest to the query "
            "vector.\n\n"
            "Vector search is useful in machine learning and natural language processing because it can find semantic "
            "similarity even when two texts use different words. For example, a query about model generalization may "
            "retrieve a passage about overfitting because the concepts are related. Approximate nearest neighbor "
            "libraries such as FAISS make vector search efficient for large collections.\n\n"
            "In hybrid retrieval, vector search is combined with lexical methods such as BM25. The vector component "
            "improves semantic recall, while the lexical component preserves exact matches for names, formulas, and "
            "keywords. This combination is common in modern RAG applications."
        ),
        "url": "https://en.wikipedia.org/wiki/Vector_database",
    },
    {
        "title": "Retrieval augmented generation",
        "text": (
            "Retrieval augmented generation, often called RAG, is a technique that connects a language model to an "
            "external knowledge source. Instead of answering only from its model parameters, the system first retrieves "
            "relevant documents or passages and then generates an answer using those passages as context.\n\n"
            "A RAG system usually contains an ingestion pipeline, a chunking strategy, embeddings, an index such as "
            "FAISS, a retriever such as BM25 or vector search, an optional reranker, and a generator. The goal is to "
            "produce answers that are more grounded, more current, and easier to trace back to sources.\n\n"
            "Self-corrective RAG adds control logic around retrieval and generation. The agent can judge whether the "
            "retrieved sources are useful, reformulate the query if the sources are weak, retry retrieval several "
            "times, and refuse to answer when the indexed dataset does not contain relevant information."
        ),
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    },
    {
        "title": "Transformer architecture",
        "text": (
            "The transformer is a neural network architecture widely used in natural language processing and modern "
            "artificial intelligence. It is based on attention mechanisms that allow the model to weigh relationships "
            "between tokens in a sequence. Transformers replaced many recurrent neural network approaches because "
            "they train efficiently and capture long-range dependencies.\n\n"
            "Self-attention lets each token attend to other tokens in the same sequence. Multi-head attention repeats "
            "this process in parallel so the model can learn different kinds of relationships. Transformers also use "
            "feedforward layers, normalization, residual connections, and positional information.\n\n"
            "Large language models such as BERT, GPT, T5, and many instruction-tuned systems are based on transformer "
            "architectures. Transformers are also used in computer vision, speech recognition, multimodal learning, "
            "retrieval, ranking, and code generation."
        ),
        "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    },
    {
        "title": "Large language model",
        "text": (
            "A large language model is a machine learning model trained on large collections of text to predict and "
            "generate language. Large language models can answer questions, summarize documents, translate text, write "
            "code, extract information, and participate in conversations. Most modern large language models use the "
            "transformer architecture.\n\n"
            "Training a language model usually involves pretraining on broad text data and then adapting the model "
            "with supervised fine-tuning, instruction tuning, preference optimization, or reinforcement learning from "
            "feedback. The model learns statistical patterns in language, but it can still produce hallucinations if "
            "it answers without enough evidence.\n\n"
            "RAG is often used with large language models to ground answers in external documents. Guardrails, source "
            "citation, verification, and refusal behavior are important when the dataset does not contain the answer."
        ),
        "url": "https://en.wikipedia.org/wiki/Large_language_model",
    },
    {
        "title": "Classification",
        "text": (
            "Classification is a supervised machine learning task where a model assigns an input example to one of "
            "several predefined categories. Examples include spam detection, sentiment analysis, image recognition, "
            "fraud detection, medical diagnosis, and document routing. A binary classifier chooses between two labels, "
            "while a multiclass classifier chooses among several labels.\n\n"
            "Classification models include logistic regression, decision trees, random forests, support vector machines, "
            "naive Bayes classifiers, nearest neighbor methods, and neural networks. The model is trained on labeled "
            "examples and evaluated on data that was not used during training.\n\n"
            "Important classification metrics include accuracy, precision, recall, F1 score, ROC AUC, and confusion "
            "matrices. The right metric depends on the cost of false positives and false negatives. Class imbalance "
            "can make accuracy misleading, so precision and recall are often more informative."
        ),
        "url": "https://en.wikipedia.org/wiki/Statistical_classification",
    },
    {
        "title": "Regression analysis",
        "text": (
            "Regression is a supervised machine learning task where a model predicts a continuous numeric value. "
            "Examples include predicting house prices, demand, temperature, risk scores, or the time required to "
            "complete a task. Linear regression is one of the simplest and most interpretable regression methods.\n\n"
            "Regression models learn relationships between input features and a target value. Common methods include "
            "linear regression, ridge regression, lasso regression, decision tree regression, random forest regression, "
            "gradient boosting, support vector regression, and neural networks.\n\n"
            "Regression is evaluated with metrics such as mean squared error, root mean squared error, mean absolute "
            "error, and R-squared. Overfitting can occur in regression when the model captures noise in the training "
            "data instead of the underlying relationship."
        ),
        "url": "https://en.wikipedia.org/wiki/Regression_analysis",
    },
    {
        "title": "Clustering",
        "text": (
            "Clustering is an unsupervised machine learning task where examples are grouped according to similarity. "
            "Unlike classification, clustering does not require predefined labels. It is used for customer segmentation, "
            "document grouping, anomaly detection, image analysis, and exploratory data analysis.\n\n"
            "K-means clustering partitions data into a chosen number of clusters by minimizing distances to cluster "
            "centers. Hierarchical clustering builds a tree of nested groups. DBSCAN finds dense regions and can "
            "identify outliers. The best clustering method depends on the geometry and scale of the data.\n\n"
            "Clustering can be applied to embeddings generated by neural networks or language models. In information "
            "retrieval, clustering helps organize documents, detect topics, and improve search or recommendation "
            "systems."
        ),
        "url": "https://en.wikipedia.org/wiki/Cluster_analysis",
    },
    {
        "title": "Word embedding",
        "text": (
            "A word embedding is a numeric representation of a word or token in a vector space. Words with related "
            "meanings tend to have vectors that are close together. Embeddings are used in natural language processing, "
            "information retrieval, recommendation systems, and retrieval augmented generation.\n\n"
            "Older embedding methods include Word2Vec, GloVe, and fastText. Modern language models produce contextual "
            "embeddings, where the representation of a word depends on the surrounding sentence. Sentence embeddings "
            "represent whole sentences or passages and are useful for semantic search.\n\n"
            "Embeddings make it possible to compare text by meaning rather than exact words. A vector index such as "
            "FAISS can store embeddings for many chunks and retrieve the chunks closest to a user query."
        ),
        "url": "https://en.wikipedia.org/wiki/Word_embedding",
    },
    {
        "title": "Model evaluation",
        "text": (
            "Model evaluation is the process of measuring how well a machine learning model performs. Evaluation uses "
            "data that was not used for training so that the results estimate generalization. A common split separates "
            "data into training, validation, and test sets.\n\n"
            "Different tasks require different metrics. Classification can use accuracy, precision, recall, F1 score, "
            "and ROC AUC. Regression can use mean squared error and mean absolute error. Ranking and retrieval can use "
            "precision at k, recall at k, mean reciprocal rank, and normalized discounted cumulative gain.\n\n"
            "Evaluation is also important for RAG systems. A RAG answer can be evaluated for relevance, groundedness, "
            "faithfulness, answer completeness, and source quality. If no reliable source is retrieved, the system "
            "should refuse rather than hallucinate an unsupported answer."
        ),
        "url": "https://en.wikipedia.org/wiki/Evaluation_of_machine_learning_models",
    },
    {
        "title": "Blockchain",
        "text": (
            "A blockchain is a distributed ledger that stores records in linked blocks. Each block usually contains "
            "transactions, a timestamp, and a cryptographic hash of the previous block. This structure makes old "
            "records difficult to change without changing later blocks. Blockchains are used in cryptocurrencies, "
            "digital assets, supply chain tracking, and decentralized applications.\n\n"
            "Blockchain is not itself a machine learning method, but it is related to data systems, security, privacy, "
            "and trustworthy computation. Machine learning can be used to detect fraud in blockchain transactions, "
            "analyze network behavior, estimate risk, and classify addresses or smart contracts.\n\n"
            "In AI systems, blockchain is sometimes discussed as a way to track data provenance, audit model usage, "
            "or coordinate decentralized computation. These applications are separate from core machine learning "
            "algorithms such as classification, regression, clustering, and neural networks."
        ),
        "url": "https://en.wikipedia.org/wiki/Blockchain",
    },
    {
        "title": "Convolutional neural network",
        "text": (
            "A convolutional neural network, or CNN, is a deep learning architecture commonly used for images, video, "
            "medical imaging, object detection, and computer vision. CNNs use convolutional filters to detect local "
            "patterns such as edges, corners, textures, shapes, and visual parts. Early layers learn simple visual "
            "features, while deeper layers combine them into more abstract representations.\n\n"
            "A CNN usually contains convolution layers, activation functions such as ReLU, pooling layers, normalization "
            "layers, and fully connected or classification heads. Pooling reduces spatial size and helps the network "
            "be more robust to small translations. Modern CNN architectures include LeNet, AlexNet, VGG, ResNet, "
            "DenseNet, EfficientNet, and YOLO-style detectors.\n\n"
            "CNNs are trained with backpropagation and gradient descent. They can overfit when the training dataset is "
            "small, so data augmentation, dropout, regularization, transfer learning, and pretrained models are often "
            "used to improve generalization."
        ),
        "url": "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    },
    {
        "title": "Recurrent neural network",
        "text": (
            "A recurrent neural network, or RNN, is a neural network architecture designed for sequential data such as "
            "text, audio, time series, and sensor signals. RNNs process one step at a time and maintain a hidden state "
            "that carries information from previous steps. This makes them useful for sequence modeling tasks.\n\n"
            "Basic RNNs can suffer from vanishing and exploding gradients when learning long-range dependencies. LSTM "
            "and GRU networks were designed to improve memory and gradient flow. Before transformers became dominant, "
            "RNNs were widely used for machine translation, speech recognition, language modeling, and sequence tagging.\n\n"
            "Transformers often outperform RNNs on large language tasks because attention allows parallel processing "
            "and direct connections between distant tokens. RNNs are still useful for some streaming, low-latency, and "
            "time-series applications."
        ),
        "url": "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    },
    {
        "title": "Attention mechanism",
        "text": (
            "An attention mechanism is a neural network technique that lets a model focus on the most relevant parts "
            "of an input. In natural language processing, attention helps a model decide which tokens matter most when "
            "representing or generating another token. Self-attention compares tokens inside the same sequence.\n\n"
            "Attention uses queries, keys, and values. A query is compared with keys to produce attention weights, and "
            "those weights are used to combine values. Multi-head attention repeats this process several times so the "
            "model can learn different relationships, such as syntax, coreference, topic, or position.\n\n"
            "Attention is the central component of transformer models. It is used in BERT, GPT, T5, vision transformers, "
            "multimodal models, rerankers, and many retrieval augmented generation systems."
        ),
        "url": "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    },
    {
        "title": "BERT",
        "text": (
            "BERT is a transformer-based language model designed to learn bidirectional contextual representations of "
            "text. Unlike left-to-right language models, BERT looks at context on both sides of a token during "
            "pretraining. It is commonly used for classification, named entity recognition, semantic similarity, "
            "question answering, retrieval, and reranking.\n\n"
            "BERT is pretrained with masked language modeling and next sentence prediction in the original formulation. "
            "During masked language modeling, some tokens are hidden and the model learns to predict them from context. "
            "Fine-tuning adapts BERT to a downstream supervised task using labeled examples.\n\n"
            "Many cross-encoder rerankers are based on BERT-like architectures. A cross-encoder reads the query and "
            "document together, making it accurate for reranking but slower than bi-encoder vector retrieval."
        ),
        "url": "https://en.wikipedia.org/wiki/BERT_(language_model)",
    },
    {
        "title": "Cross-encoder and bi-encoder",
        "text": (
            "A bi-encoder is a retrieval model that encodes a query and a document separately into embeddings. Similarity "
            "is computed with cosine similarity, dot product, or another vector score. Bi-encoders are efficient because "
            "document embeddings can be precomputed and stored in a vector index such as FAISS.\n\n"
            "A cross-encoder is a reranking model that reads the query and document together in one transformer input. "
            "Because it can attend across both texts, a cross-encoder often gives more accurate relevance scores than a "
            "bi-encoder. The tradeoff is speed: every query-document pair must be evaluated at runtime.\n\n"
            "Modern RAG pipelines often use both methods. A BM25 or bi-encoder retriever quickly finds candidate chunks, "
            "then a cross-encoder reranker reorders the best candidates before answer generation."
        ),
        "url": "https://www.sbert.net/examples/cross_encoder/applications/README.html",
    },
    {
        "title": "Chunking strategy",
        "text": (
            "Chunking is the process of splitting documents into smaller passages before indexing them for retrieval. "
            "RAG systems use chunking because language models and retrievers work better with focused passages than "
            "very long documents. A chunk should be large enough to preserve context but small enough to stay relevant.\n\n"
            "Common chunking parameters include chunk size, chunk overlap, minimum chunk length, separators, and metadata. "
            "Overlap repeats a small part of the previous chunk so important information is not lost at boundaries. "
            "Metadata such as title, URL, section name, and document id helps with source display and filtering.\n\n"
            "Bad chunking can hurt retrieval. Chunks that are too large mix unrelated ideas, while chunks that are too "
            "small lose context. Semantic chunking, sentence splitting, and section-aware splitting can improve source "
            "quality in production RAG systems."
        ),
        "url": "https://en.wikipedia.org/wiki/Information_retrieval",
    },
    {
        "title": "FAISS",
        "text": (
            "FAISS is a library for efficient similarity search and clustering of dense vectors. It is often used to "
            "build vector indexes for machine learning, semantic search, recommendation systems, and retrieval augmented "
            "generation. FAISS can search millions of embeddings faster than a brute-force comparison in many settings.\n\n"
            "A simple FAISS index such as IndexFlatIP performs exact inner product search. Other index types use "
            "approximate nearest neighbor methods, quantization, or inverted files to improve speed and memory usage. "
            "The best index depends on dataset size, latency requirements, memory constraints, and desired recall.\n\n"
            "In a RAG pipeline, text chunks are embedded with an embedding model, stored in a FAISS index, and searched "
            "with an embedded user query. The retrieved chunks become context for the generator or are passed to a "
            "reranker before generation."
        ),
        "url": "https://en.wikipedia.org/wiki/Nearest_neighbor_search",
    },
    {
        "title": "Loss function",
        "text": (
            "A loss function measures how wrong a machine learning model is on a training example or batch. Training "
            "tries to minimize the loss by updating model parameters. The choice of loss function depends on the task, "
            "such as classification, regression, ranking, language modeling, or contrastive learning.\n\n"
            "Cross-entropy loss is common for classification and language modeling. Mean squared error and mean absolute "
            "error are common for regression. Contrastive loss and triplet loss are used to train embedding models that "
            "place similar examples close together and dissimilar examples far apart.\n\n"
            "A lower training loss does not always mean a better model. If validation loss increases while training loss "
            "continues to decrease, the model may be overfitting. Regularization, early stopping, and better data can "
            "improve generalization."
        ),
        "url": "https://en.wikipedia.org/wiki/Loss_function",
    },
    {
        "title": "Gradient descent and backpropagation",
        "text": (
            "Gradient descent is an optimization algorithm used to train machine learning models. It updates parameters "
            "in the direction that reduces the loss function. Stochastic gradient descent updates parameters with small "
            "batches of data, while batch gradient descent uses the full training set.\n\n"
            "Backpropagation is the algorithm used to compute gradients in neural networks. It applies the chain rule "
            "from calculus to determine how each parameter contributed to the final error. The optimizer then uses "
            "those gradients to update weights.\n\n"
            "Important training concepts include learning rate, batch size, epochs, momentum, Adam, weight decay, "
            "gradient clipping, vanishing gradients, exploding gradients, and convergence. Choosing these settings is "
            "part of hyperparameter tuning."
        ),
        "url": "https://en.wikipedia.org/wiki/Backpropagation",
    },
    {
        "title": "Regularization and dropout",
        "text": (
            "Regularization is a set of techniques used to reduce overfitting in machine learning. It discourages a "
            "model from becoming too complex or too dependent on noise in the training data. L1 regularization can "
            "encourage sparse weights, while L2 regularization or weight decay penalizes large weights.\n\n"
            "Dropout is a regularization method for neural networks. During training, it randomly disables some neurons "
            "or activations, forcing the network to learn more robust representations. At inference time, the full "
            "network is used. Dropout is common in feedforward networks, CNNs, and transformer models.\n\n"
            "Other regularization techniques include data augmentation, early stopping, label smoothing, pruning, "
            "batch normalization, simpler model architectures, and collecting more diverse training data."
        ),
        "url": "https://en.wikipedia.org/wiki/Regularization_(mathematics)",
    },
    {
        "title": "Fine-tuning and transfer learning",
        "text": (
            "Transfer learning uses knowledge learned from one task or dataset to improve performance on another task. "
            "In deep learning, a model is often pretrained on a large general dataset and then fine-tuned on a smaller "
            "task-specific dataset. This reduces the amount of labeled data needed for good performance.\n\n"
            "Fine-tuning updates a pretrained model's parameters using examples from the target task. In NLP, language "
            "models can be fine-tuned for classification, extraction, summarization, instruction following, or retrieval. "
            "In computer vision, CNNs or vision transformers can be fine-tuned for specific image categories.\n\n"
            "Parameter-efficient fine-tuning methods such as adapters and LoRA update only a small number of additional "
            "parameters. This can reduce memory, storage, and compute costs while adapting large models."
        ),
        "url": "https://en.wikipedia.org/wiki/Transfer_learning",
    },
    {
        "title": "Generative adversarial network",
        "text": (
            "A generative adversarial network, or GAN, is a generative deep learning framework with two models: a "
            "generator and a discriminator. The generator creates synthetic examples, while the discriminator tries to "
            "distinguish generated examples from real training examples. Training is adversarial because the two models "
            "compete against each other.\n\n"
            "GANs have been used for image synthesis, style transfer, super-resolution, data augmentation, and domain "
            "adaptation. A well-trained generator can produce realistic samples, but GAN training can be unstable and "
            "may suffer from mode collapse, where the generator produces limited varieties of outputs.\n\n"
            "Diffusion models have replaced GANs in many modern image generation systems, but GANs remain an important "
            "concept in generative AI and representation learning."
        ),
        "url": "https://en.wikipedia.org/wiki/Generative_adversarial_network",
    },
    {
        "title": "Diffusion model",
        "text": (
            "A diffusion model is a generative model that learns to create data by reversing a gradual noising process. "
            "During training, noise is added to examples step by step, and the model learns how to denoise them. During "
            "generation, the model starts from noise and iteratively produces a sample.\n\n"
            "Diffusion models are widely used for image generation, audio generation, video generation, and multimodal "
            "AI. Text-to-image systems use a text encoder to condition the denoising process on a prompt. Classifier-free "
            "guidance and latent diffusion are common techniques in practical systems.\n\n"
            "Compared with GANs, diffusion models are often more stable to train and can produce diverse, high-quality "
            "samples, though generation may require many denoising steps unless accelerated."
        ),
        "url": "https://en.wikipedia.org/wiki/Diffusion_model",
    },
    {
        "title": "Reinforcement learning",
        "text": (
            "Reinforcement learning is a machine learning paradigm where an agent learns to choose actions by interacting "
            "with an environment. The agent receives rewards or penalties and tries to learn a policy that maximizes "
            "expected cumulative reward. It is used in robotics, games, control systems, recommendation, and alignment.\n\n"
            "Important reinforcement learning concepts include state, action, reward, policy, value function, Q-learning, "
            "temporal difference learning, exploration, exploitation, and Markov decision processes. Deep reinforcement "
            "learning combines neural networks with reinforcement learning algorithms.\n\n"
            "Reinforcement learning from human feedback, or RLHF, is used to align large language models with human "
            "preferences. Preference data is used to train a reward model, and the language model is optimized to produce "
            "responses preferred by humans."
        ),
        "url": "https://en.wikipedia.org/wiki/Reinforcement_learning",
    },
    {
        "title": "Data preprocessing and feature engineering",
        "text": (
            "Data preprocessing prepares raw data for machine learning. It can include cleaning missing values, removing "
            "duplicates, handling outliers, normalizing numeric features, encoding categorical variables, tokenizing text, "
            "resizing images, and splitting data into training, validation, and test sets.\n\n"
            "Feature engineering creates useful input variables for a model. In traditional machine learning, good "
            "features can strongly improve performance. Examples include TF-IDF features for text, date features for "
            "time series, aggregations for tabular data, and domain-specific signals.\n\n"
            "Deep learning often learns representations automatically, but preprocessing still matters. Poor data quality "
            "can cause biased models, unstable training, leakage, overfitting, and unreliable evaluation."
        ),
        "url": "https://en.wikipedia.org/wiki/Feature_engineering",
    },
    {
        "title": "Confusion matrix and metrics",
        "text": (
            "A confusion matrix summarizes classification predictions by counting true positives, false positives, true "
            "negatives, and false negatives. It helps diagnose which classes a model confuses and whether errors are "
            "balanced or concentrated in particular categories.\n\n"
            "Precision measures how many predicted positives are actually positive. Recall measures how many real "
            "positives were found by the model. F1 score is the harmonic mean of precision and recall. Accuracy measures "
            "the total fraction of correct predictions but can be misleading when classes are imbalanced.\n\n"
            "Choosing the right metric depends on the application. In medical diagnosis or fraud detection, false "
            "negatives and false positives can have very different costs, so precision, recall, F1, and ROC AUC are "
            "often more informative than accuracy alone."
        ),
        "url": "https://en.wikipedia.org/wiki/Confusion_matrix",
    },
]


def sample_articles() -> list[dict[str, str]]:
    return [{**article, "id": slug(article["title"])} for article in SAMPLE_ARTICLES]
