# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes

import sys
import string
from nltk.corpus import stopwords
import re
from random import shuffle, seed
from unittest import TestCase, main, skip

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return {'even_odd': str((self.data % 2))}

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""

        regex = re.compile('[^a-zA-Z \']')
        self.data = regex.sub('', self.data.lower())

        words = self.data.split()

        """
        # STOPWORDS
        cachedStopWords = stopwords.words("english")
        filtered_words = [word for word in words if word not in cachedStopWords]

        # BIGRAMS
        bigrams = []
        prev = ""
        =for word in words:
            if (prev != ""):
                bigrams.append(prev + " " + word)
            prev = word
        
        # NUM OF PUNCTUATIONS
        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        punctuation = str(count(self.data, string.punctuation))
        
        # WORD LENGTH
        word_length = []
        for word in words:
            word_length.append(str(count(word, string.ascii_letters)))
        """

        return {'bag_of_words': words}

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return {'first_letter': [name[0].lower()], 'last_letter': [name[-1].lower()], 'characters': list(name)}

def compute_accuracy(truth, test, verbose=sys.stderr):
    correct = [truth[i].label == test[i] for i in range(0, len(truth))]
    if verbose:
        print("Accuracy: {:.2f}% ".format(100 * sum(correct) / len(correct)), file=verbose)
    return sum(correct) / len(correct)

def compute_precision(truth, test, verbose=sys.stderr):
    labels = (x.label for x in truth)
    precision = {}
    for label in labels:
        tp = sum([test[i] == label and truth[i].label == label for i in range(0, len(truth))])
        fp = sum([test[i] == label and truth[i].label != label for i in range(0, len(truth))])
        if (tp + fp == 0):
            print("There are no instances that have been classified as: ", label)
            precision[label] = 0
        else:
            precision[label] = tp / (tp + fp)

    if verbose:
        print("\n".join('Precision for {}: {:.2f}%'.format(label, (100 * precision[label])) for label in precision), file=verbose)
    return precision

def compute_recall(truth, test, verbose=sys.stderr):
    labels = (x.label for x in truth)
    recall = {}
    for label in labels:
        tp = sum([truth[i].label == label and test[i] == label for i in range(0, len(truth))])
        fn = sum([truth[i].label == label and test[i] != label for i in range(0, len(truth))])
        if (tp + fn == 0):
            print("There are no instances that have been classified as: ", label)
            recall[label] = 0
        else:
            recall[label] = tp / (tp + fn)
    if verbose:
        print("\n".join('Recall for {}: {:.2f}%'.format(label, (100 * recall[label])) for label in recall), file=verbose)
    return recall

def compute_fscore(truth, test, verbose=sys.stderr):
    labels = (x.label for x in truth)
    precision = compute_precision(truth, test, False)
    recall = compute_recall(truth, test, False)
    fscore = {}
    for label in precision:
        if (precision[label] + recall[label] == 0):
            fscore[label] = 0
        else:
            fscore[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
    if verbose:
        print("\n".join('Fscore for {}: {:.2f}%'.format(label, (100 * fscore[label])) for label in fscore), file=verbose)
    return fscore

def compute_all_stats(truth, test, verbose=sys.stderr):
    accuracy = compute_accuracy(truth, test)
    compute_precision(truth, test)
    compute_recall(truth, test)
    compute_fscore(truth, test)

    return accuracy

def classify(classifier, test):
    return [classifier.classify(x) for x in test]

class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd(self):
        """Classify numbers as even or odd"""
        classifier = NaiveBayes()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        classified = classify(classifier, test)
        self.assertEqual(compute_all_stats(test, classified), 1.0)

    def split_names_corpus(self, document_class=Name):
        # Split the names corpus into training and test sets
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:6000], names[6000:])

    def test_names_nltk(self):
        # Classify names using NLTK features
        train, test = self.split_names_corpus()
        classifier = NaiveBayes()
        classifier.train(train)
        classified = classify(classifier, test)
        self.assertGreater(compute_all_stats(test, classified), 0.70)

    def split_blogs_corpus(self, document_class):
        # Split the blog post corpus into training and test sets
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag(self):
        # Classify blog authors using bag-of-words
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        classified = classify(classifier, test)
        self.assertGreater(compute_all_stats(test, classified), 0.55)

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba(self):
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        # you don't need to pass this test
        classified = classify(classifier, test)
        self.assertGreater(compute_all_stats(test, classified), 0.1)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
