r"""Compute active speaker detection performance for the AVA dataset.
Please send any questions about this code to the Google Group ava-dataset-users:
https://groups.google.com/forum/#!forum/ava-dataset-users
Example usage:
python -O get_ava_active_speaker_performance.py \
-g testdata/eval.csv \
-p testdata/predictions.csv \
-v
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.
  Precision is modified to ensure that it does not decrease as recall
  decrease.
  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls
  Raises:
    ValueError: if the input is not of the correct format
  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.
  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Smooth precision to be monotonically decreasing.
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def load_csv(filename, column_names):
  """Loads CSV from the filename using given column names.
  Adds uid column.
  Args:
    filename: Path to the CSV file to load.
    column_names: A list of column names for the data.
  Returns:
    df: A Pandas DataFrame containing the data.
  """
  # Here and elsewhere, df indicates a DataFrame variable.

  df = pd.read_csv(filename, usecols=column_names)
  #df = pd.read_csv(filename, header=None, names=column_names)
  
  # Creates a unique id from frame timestamp and entity id.
  df["uid"] = (df["frame_timestamp"].map(str) + ":" + df["entity_id"])  
  return df


def eq(a, b, tolerance=1e-09):
  """Returns true if values are approximately equal."""
  return abs(a - b) <= tolerance


def merge_groundtruth_and_predictions(df_groundtruth, df_predictions):
  """Merges groundtruth and prediction DataFrames.
  The returned DataFrame is merged on uid field and sorted in descending order
  by score field. Bounding boxes are checked to make sure they match between
  groundtruth and predictions.
  Args:
    df_groundtruth: A DataFrame with groundtruth data.
    df_predictions: A DataFrame with predictions data.
  Returns:
    df_merged: A merged DataFrame, with rows matched on uid column.
  """
  if df_groundtruth["uid"].count() != df_predictions["uid"].count():
    raise ValueError(
        "Groundtruth and predictions CSV must have the same number of "
        "unique rows.")
  # print(df_predictions["label"].unique())
  if df_predictions["label"].unique() != ["SPEAKING_AUDIBLE"]:
    raise ValueError(
        "Predictions CSV must contain only SPEAKING_AUDIBLE label.")

  if df_predictions["score"].count() < df_predictions["uid"].count():
    raise ValueError("Predictions CSV must contain score value for every row.")

  # Merges groundtruth and predictions on uid, validates that uid is unique
  # in both frames, and sorts the resulting frame by the predictions score.
  df_merged = df_groundtruth.merge(
      df_predictions,
      on="uid",
      suffixes=("_groundtruth", "_prediction"),
      validate="1:1").sort_values(
          by=["score"], ascending=False).reset_index()
  # Validates that bounding boxes in ground truth and predictions match for the
  # same uids.
  df_merged["bounding_box_correct"] = np.where(
      eq(df_merged["entity_box_x1_groundtruth"],
         df_merged["entity_box_x1_prediction"])
      & eq(df_merged["entity_box_x2_groundtruth"],
           df_merged["entity_box_x2_prediction"])
      & eq(df_merged["entity_box_y1_groundtruth"],
           df_merged["entity_box_y1_prediction"])
      & eq(df_merged["entity_box_y2_groundtruth"],
           df_merged["entity_box_y2_prediction"]), True, False)

  if (~df_merged["bounding_box_correct"]).sum() > 0:
    raise ValueError(
        "Mismatch between groundtruth and predictions bounding boxes found at "
        + str(list(df_merged[~df_merged["bounding_box_correct"]]["uid"])))

  return df_merged


def get_all_positives(df_merged):
  """Counts all positive examples in the groundtruth dataset."""
  return df_merged[df_merged["label_groundtruth"] ==
                   "SPEAKING_AUDIBLE"]["uid"].count()


def calculate_precision_recall(df_merged):
  """Calculates precision and recall arrays going through df_merged row-wise."""
  all_positives = get_all_positives(df_merged)
  # Populates each row with 1 if this row is a true positive
  # (at its score level).
  df_merged["is_tp"] = np.where(
      (df_merged["label_groundtruth"] == "SPEAKING_AUDIBLE") &
      (df_merged["label_prediction"] == "SPEAKING_AUDIBLE"), 1, 0)

  # Counts true positives up to and including that row.
  df_merged["tp"] = df_merged["is_tp"].cumsum()

  # Calculates precision for every row counting true positives up to
  # and including that row over the index (1-based) of that row.
  df_merged["precision"] = df_merged["tp"] / (df_merged.index + 1)
  # Calculates recall for every row counting true positives up to
  # and including that row over all positives in the groundtruth dataset.

  df_merged["recall"] = df_merged["tp"] / all_positives
  logging.info(
      "\n%s\n",
      df_merged.head(10)[[
          "uid", "score", "label_groundtruth", "is_tp", "tp", "precision",
          "recall"
      ]])

  return np.array(df_merged["precision"]), np.array(df_merged["recall"])


def run_evaluation(groundtruth, predictions):
  """Runs AVA Active Speaker evaluation, printing average precision result."""
  df_groundtruth = load_csv(
      groundtruth,
      column_names=[
          "video_id", "frame_timestamp", "entity_box_x1", "entity_box_y1",
          "entity_box_x2", "entity_box_y2", "label", "entity_id"
      ])
  df_predictions = load_csv(
      predictions,
      column_names=[
          "video_id", "frame_timestamp", "entity_box_x1", "entity_box_y1",
          "entity_box_x2", "entity_box_y2", "label", "entity_id", "score"
      ])
  df_merged = merge_groundtruth_and_predictions(df_groundtruth, df_predictions)
  precision, recall = calculate_precision_recall(df_merged)
  mAP = 100 * compute_average_precision(precision, recall)
  print("average precision: %2.2f%%"%(mAP))
  return mAP


def parse_arguments():
  """Parses command-line flags.
  Returns:
    args: a named tuple containing three file objects args.labelmap,
    args.groundtruth, and args.detections.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-g",
      "--groundtruth",
      help="CSV file containing ground truth.",
      type=argparse.FileType("r"),
      required=True)
  parser.add_argument(
      "-p",
      "--predictions",
      help="CSV file containing active speaker predictions.",
      type=argparse.FileType("r"),
      required=True)
  parser.add_argument(
      "-v", "--verbose", help="Increase output verbosity.", action="store_true")
  return parser.parse_args()


def main():
  start = time.time()
  args = parse_arguments()
  if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
  del args.verbose
  mAP = run_evaluation(**vars(args))
  logging.info("Computed in %s seconds", time.time() - start)
  return mAP

if __name__ == "__main__":
  main()