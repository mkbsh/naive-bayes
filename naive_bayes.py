# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from collections import defaultdict
from math import log

class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self, model={}):
        super(NaiveBayes, self).__init__(model)
        self.prior = None

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)
    def get_labels(self): return list(label for label in self.prior)

    def train(self, instances):
        """Remember the labels associated with the features of instances."""

        # Count documents with label: {label1: 2, label2: 4, ...}
        # Used to approximate the prior.
        prior = defaultdict()

        # First, this contains the conditional probability of feature given class P( X(i)|Y(j) ):
        #   {label1: {category1: {feature1: 0.4, feature2: 0.3}}, {category2: {feature1: 0.6}}}
        # Later, smoothed and converted to log probabilities to prevent underflow errors.
        likelihood = defaultdict()

        # Distinct features aggregated across all labels in category. Used for smoothing.
        category_features = defaultdict()

        for instance in instances:
            if (instance.label == ""):
                continue
            try:
                prior[instance.label] += 1
            except KeyError:
                prior[instance.label] = 1
                likelihood[instance.label] = {}
            for category, features in instance.features().items():
                if category not in likelihood[instance.label].keys():
                    likelihood[instance.label][category] = {}
                if category not in category_features.keys():
                    category_features[category] = set()
                for feature in features:
                    
                    try:
                        likelihood[instance.label][category][feature] += 1
                    except KeyError:
                        likelihood[instance.label][category][feature] = 1
                        category_features[category].add(feature)
                    """
                    likelihood[instance.label][category][feature] = 1
                    category_features[category].add(feature)
                    """

        for label in prior:
            prior[label] = log(float(prior[label]) / len(instances))

        # P(feature A | label X) = (count of feature A in label X) / (count of all features in label X)
        # Apply Laplace smoothing:
        # P(feature A | label X) = (count of feature A in label X) + 1 /
        #                                   {(count of all features in label X) + (number of distinct features) + 1}
        # It follows that if a feature is not present in label X but is a known feature:
        # P(feature A | label X) = 1 / {(count of all features in label X) + (number of distinct features) + 1}
        # Include this in "UNKNOWN" feature in the model to be used as a penalty in classification step.
        for label, all_features in likelihood.items():
            for category, features in all_features.items():
                sum_label = sum(likelihood[label][category].values())
                denom = sum_label + len(category_features[category]) + 1
                for feature in features.keys():
                    likelihood[label][category][feature] = log(float(likelihood[label][category][feature] + 1) / denom)
                likelihood[label][category]["UNKNOWN"] = log(float(1) / denom)

        self.model = likelihood
        self.prior = prior

    def classify(self, instance):
        """Classify an instance and return the expected label."""
        prob = defaultdict()
        for label in self.model:
            prob[label] = 0
            for category, features in instance.features().items():
                for feature in features:
                    if feature in self.model[label][category]:
                        prob[label] += self.model[label][category][feature]
                    else:
                        prob[label] += self.model[label][category]["UNKNOWN"]
            prob[label] += self.prior[label] if self.prior is not None else 0

        return max(prob, key=prob.get)
