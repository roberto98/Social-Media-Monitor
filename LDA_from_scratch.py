
import numpy as np
import random
from tqdm import tqdm


class LDA:
    def __init__(self, num_topics, num_iterations=50, id2word=None, alpha=1, beta=1):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.topic_word_counts = None
        self.doc_topic_counts = None
        self.id2word = id2word

    def fit(self, corpus):
        """
        Fit the LDA model to the given corpus.

        Parameters:
        - corpus (list of lists): The corpus of documents, where each document is represented as a list of (word, count) tuples.

        """
        self.corpus = corpus
        self.num_words = max([word_idx for doc in corpus for word_idx, _ in doc]) + 1
        self.initialize()
        self.sample_topics()

    def initialize(self):
        """
        Initialize the topic assignments and count matrices.

        """
        # Initialize topic assignments randomly
        self.topic_assignments = [[random.randint(0, self.num_topics - 1) for _ in doc] for doc in self.corpus]

        # Initialize topic-word and document-topic count matrices
        self.topic_word_counts = np.zeros((self.num_topics, self.num_words))
        self.doc_topic_counts = np.zeros((len(self.corpus), self.num_topics))

        # Count initial topic assignments
        for doc_idx, doc in enumerate(self.corpus):
            for word_idx, (word, count) in enumerate(doc):
                topic = self.topic_assignments[doc_idx][word_idx]
                self.topic_word_counts[topic][word] += count
                self.doc_topic_counts[doc_idx][topic] += count

    def sample_topics(self):
        """
        Estimate the topic assignments computing the probabilities.

        """
        for it in tqdm(range(self.num_iterations), desc="Training"):
            doc_topic_sums = self.doc_topic_counts.sum(axis=1)
            topic_word_sums = self.topic_word_counts.sum(axis=1)
            
            for doc_idx, doc in enumerate(self.corpus):
                for word_idx, (word, count) in enumerate(doc):
                    # Remove current topic assignment
                    old_topic = self.topic_assignments[doc_idx][word_idx]
                    self.topic_word_counts[old_topic][word] -= count
                    self.doc_topic_counts[doc_idx][old_topic] -= count
                    doc_topic_sums[doc_idx] -= count
                    topic_word_sums[old_topic] -= count

                    # Compute probabilities for each topic 
                    p_topic_given_doc = (self.doc_topic_counts[doc_idx, :] + self.alpha) / (doc_topic_sums[doc_idx] + self.num_topics * self.alpha)
                    p_word_given_topic = (self.topic_word_counts[:, word] + self.beta) / (topic_word_sums + self.num_words * self.beta)
                    probabilities = p_topic_given_doc * p_word_given_topic

                    # Normalize probabilities
                    probabilities /= probabilities.sum()

                    # Sample a new topic assignment
                    new_topic = np.random.choice(self.num_topics, p=probabilities)
                    self.topic_assignments[doc_idx][word_idx] = new_topic

                    # Update counts for new topic assignment
                    self.topic_word_counts[new_topic][word] += count
                    self.doc_topic_counts[doc_idx][new_topic] += count
                    doc_topic_sums[doc_idx] += count
                    topic_word_sums[new_topic] += count

    def get_top_words(self, num_words):
        """
        Get the top words for each topic in the LDA model.

        Parameters:
        - num_words (int): The number of top words to retrieve for each topic.

        Returns:
        - top_words (list of lists): The top words for each topic.
        """
        top_words = []
        for topic_id in range(self.num_topics):
            top_words.append(self.show_topic(topic_id, num_words))
        return top_words

    def get_topics(self):
        """
        Get the topic-word distribution matrix.

        Parameters:
        None

        Returns:
        - topics (numpy array): The topic-word distribution matrix.
        """
        topics = np.zeros((self.num_topics, self.num_words))
        for topic_idx in range(self.num_topics):
            topic_word_counts = self.topic_word_counts[topic_idx, :]
            topic_word_probs = topic_word_counts / topic_word_counts.sum()
            topics[topic_idx, :] = topic_word_probs
        return topics  

    def show_topics(self, num_topics=None, num_words=10):
        """
        Show the top words for each topic in the LDA model.

        Parameters:
        - num_topics (int): The number of topics to show. If None, show all topics.
        - num_words (int): The number of top words to show for each topic.

        Returns:
        - topics (list of tuples): The top words for each topic, along with their probabilities.
        """
        if num_topics is None:
            num_topics = self.num_topics

        topics = []
        for topic_idx in range(num_topics):
            topic_word_probs = self.topic_word_counts[topic_idx, :]
            topic_word_probs /= topic_word_probs.sum()
            top_word_indices = np.argsort(topic_word_probs)[-num_words:]
            topic_words = [(self.id2word[word_idx], topic_word_probs[word_idx]) for word_idx in top_word_indices]
            topics.append((topic_idx, topic_words))
        return topics

    def show_topic(self, topic_id, num_words=10):
        """
        Show the top words for a specific topic in the LDA model.

        Parameters:
        - topic_id (int): The ID of the topic.
        - num_words (int): The number of top words to show.

        Returns:
        - topic_words (list of tuples): The top words for the specified topic, along with their probabilities.
        """
        topic_word_probs = self.topic_word_counts[topic_id, :]
        topic_word_probs /= topic_word_probs.sum()
        top_word_indices = np.argsort(topic_word_probs)[-num_words:]
        topic_words = [(self.id2word[word_idx], np.round(topic_word_probs[word_idx], 9)) for word_idx in top_word_indices]
        return topic_words

    def get_document_topics(self, unseen_document):
        """
        Get the topic distribution for a new, unseen document.

        Parameters:
        - unseen_document (list of tuples): The unseen document, represented as a list of (word, count) tuples.

        Returns:
        - inferred_topics (list of tuples): The inferred topic distribution for the unseen document.
        """
        unseen_corpus = [unseen_document]
        inferred_topics = self.inference(unseen_corpus)
        formatted_topics = [(topic_idx, np.round(prob, 9)) for topic_idx, prob in inferred_topics[0]]

        return formatted_topics

    """
    # Perform inference on a new, unseen corpus using Variational inference
    # https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
    import scipy.special as sp
    def inference(self, unseen_corpus):
        inferred_topics = []
        # Set the variational distribution parameters
        gamma = np.ones((len(unseen_corpus), self.num_topics)) * (self.alpha + float(self.num_words) / self.num_topics)
        phi = np.ones((self.num_topics, self.num_words)) / self.num_topics

        for doc_idx, doc in enumerate(unseen_corpus):
            # Initialize gamma and phi for the current document
            gamma_d = gamma[doc_idx, :].copy()
            phi_d = phi.copy()

            for it in tqdm(range(self.num_iterations), desc="Variational Sampling"):
                for word_idx, (word, count) in enumerate(doc):
                    # E-step: Update phi and gamma
                    phi_d[:, word] = self.topic_word_counts[:, word] * np.exp(sp.psi(gamma_d))
                    phi_d[:, word] /= np.sum(phi_d[:, word])
                    gamma_d = self.alpha + np.sum(phi_d, axis=1)

            # Append the topic distribution for the current document to the list
            normalized_gamma = gamma_d / np.sum(gamma_d)
            inferred_topics.append([(idx, prob) for idx, prob in enumerate(normalized_gamma)])

        return inferred_topics    
    """

    def inference(self, unseen_corpus):
        """
        Perform inference on a new, unseen corpus using Gibbs sampling.
        https://coli-saar.github.io/cl19/materials/darling-lda.pdf

        Parameters:
        - unseen_corpus (list of lists): The unseen corpus of documents, where each document is represented as a list of (word, count) tuples.

        Returns:
        - inferred_topics (list of lists): The inferred topic distributions for the unseen corpus.
        """
        inferred_topics = []

        # Initialize topic assignments and counts
        topic_assignments = [[random.randint(0, self.num_topics - 1) for _ in doc] for doc in unseen_corpus]
        doc_topic_counts = np.zeros((len(unseen_corpus), self.num_topics))
        topic_word_counts = np.zeros((self.num_topics, self.num_words))
        topic_counts = np.zeros(self.num_topics)

        # Count initial topic assignments
        for doc_idx, doc in enumerate(unseen_corpus):
            for word_idx, (word, count) in enumerate(doc):
                topic = topic_assignments[doc_idx][word_idx]
                doc_topic_counts[doc_idx][topic] += count
                topic_word_counts[topic][word] += count
                topic_counts[topic] += count

        for _ in range(self.num_iterations):
        #for it in tqdm(range(100), desc="Gibbs Sampling"):
            for doc_idx, doc in enumerate(unseen_corpus):
                for word_idx, (word, count) in enumerate(doc):
                    old_topic = topic_assignments[doc_idx][word_idx]

                    # Remove current topic assignment
                    doc_topic_counts[doc_idx][old_topic] -= count
                    topic_word_counts[old_topic][word] -= count
                    topic_counts[old_topic] -= count

                    # Compute probabilities for each topic
                    p_topic = (doc_topic_counts[doc_idx, :] + self.alpha) * (topic_word_counts[:, word] + self.beta) / (topic_counts + self.num_words * self.beta)
                    p_topic /= p_topic.sum()

                    # Sample a new topic assignment
                    new_topic = np.random.choice(self.num_topics, p=p_topic)
                    topic_assignments[doc_idx][word_idx] = new_topic

                    # Update counts for new topic assignment
                    doc_topic_counts[doc_idx][new_topic] += count
                    topic_word_counts[new_topic][word] += count
                    topic_counts[new_topic] += count

        # Compute the topic distribution for each document
        for doc_idx, doc in enumerate(unseen_corpus):
            topic_distribution = (doc_topic_counts[doc_idx, :] + self.alpha) / (np.sum(doc_topic_counts[doc_idx, :]) + self.num_topics * self.alpha)            
            formatted_distribution = [(topic_idx, np.round(prob, 9)) for topic_idx, prob in enumerate(topic_distribution)]
            inferred_topics.append(formatted_distribution)
    
        return inferred_topics