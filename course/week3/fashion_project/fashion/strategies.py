import torch
import numpy as np
from typing import List

from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  indices = []
  # ================================
  # FILL ME OUT
  # Randomly pick a 1000 examples to label. This serves as a baseline.
  # Note that we fixed the random seed above. Please do not edit.
  # HINT: when you randomly sample, do not choose duplicates.
  # HINT: please ensure indices is a list of integers

  permuted_indices = torch.randperm(budget)
  indices = permuted_indices[:budget].tolist()

  # ================================
  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  chance_prob = 1 / 10.  # may be useful
  # ================================
  # FILL ME OUT
  # Sort indices by the predicted probabilities and choose the 1000 examples with 
  # the least confident predictions. Think carefully about what "least confident" means 
  # for a N-way classification problem.
  # Take the first 1000.
  # HINT: please ensure indices is a list of integers

  # Least confident will be a probability closest to random chance: 1/N
  # Subtract this from the tensor, take the absolute value of the tensor and then sort it ascending to get the ones closest to random chance
  abs_probs = torch.abs(pred_probs - chance_prob)
  min_probs, _ = torch.min(abs_probs, dim = 1)
  _, sorted_indices = torch.sort(min_probs)
  indices[:] = sorted_indices[:budget].tolist()

  # ================================
  return indices

def margin_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Sort indices by the different in predicted probabilities in the top two classes per example.
  # Take the first 1000.

  topk_values, _ = torch.topk(pred_probs, 2, dim=1)
  margins = topk_values[:, 0] - topk_values[:, 1]
  _, sorted_indices = torch.sort(margins)
  indices[:] = sorted_indices[:budget].tolist()

  # ================================
  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  epsilon = 1e-6
  # ================================
  # FILL ME OUT
  # Entropy is defined as -E_classes[log p(class | input)] aja the expected log probability
  # over all K classes. See https://en.wikipedia.org/wiki/Entropy_(information_theory).
  # Sort the indices by the entropy of the predicted probabilities from high to low.
  # Take the first 1000.
  # HINT: Add epsilon when taking a log for entropy computation

  entropy = -torch.sum(pred_probs * np.log(pred_probs + epsilon), dim = 1)
  _, sorted_indices = torch.sort(entropy, descending = True)
  indices[:] = sorted_indices[:budget].tolist()

  # ================================
  return indices
