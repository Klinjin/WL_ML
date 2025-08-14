# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import numpy as np
from datetime import datetime as dt


class Ingestion:
    """
    Class for handling the ingestion process.

    Args:
        None

    Attributes:
        * start_time (datetime): The start time of the ingestion process.
        * end_time (datetime): The end time of the ingestion process.
        * model (object): The model object.
        * train_data (dict): The train data dict.
        * test_data (dict): The test data dict.
    """

    def __init__(self):
        """
        Initialize the Ingestion class.

        """
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_data = None
        self.test_data = None
        self.means = None
        self.errorbars = None

    def start_timer(self):
        """
        Start the timer for the ingestion process.
        """
        self.start_time = dt.now()

    def stop_timer(self):
        """
        Stop the timer for the ingestion process.
        """
        self.end_time = dt.now()

    def get_duration(self):
        """
        Get the duration of the ingestion process.

        Returns:
            timedelta: The duration of the ingestion process.
        """
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stopped. Returning None")
            return None

        return self.end_time - self.start_time

    def save_duration(self, output_dir=None):
        """
        Save the duration of the ingestion process to a file.

        Args:
            output_dir (str): The output directory to save the duration file.
        """
        duration = self.get_duration()
        duration_in_mins = int(duration.total_seconds() / 60)
        duration_file = os.path.join(output_dir, "ingestion_duration.json")
        if duration is not None:
            with open(duration_file, "w") as f:
                f.write(json.dumps({"ingestion_duration": duration_in_mins}, indent=4))

    def load_train_and_test_data(self, input_dir):
        """
        Load the training and testing data.

        """
        print("[*] Loading Train data")

        mask_file = os.path.join(input_dir, "WIDE12H_bin2_2arcmin_mask.npy")
        kappa_file = os.path.join(input_dir, "WIDE12H_bin2_2arcmin_kappa.npy")
        labels_file = os.path.join(input_dir, "label.npy")

        shape = [1424, 176]

        mask = np.load(mask_file)
        kappa = np.zeros((101, 256, *shape), dtype=np.float16)

        # This is the train data; shape = (101, 256, 1424, 176)
        # 101 = realizations of 2 parameters of interest
        # 256 = realizations of 3 nuisance parameters
        # (1424, 176) = image dimension
        kappa[:, :, mask] = np.load(kappa_file)

        # Train label shape = (101, 256, 2 POIs + 3 predefined NPs) = (101, 256, 5)
        labels = np.load(labels_file)

        self.train_data = {
            "data": kappa,
            "labels": labels
        }

        print("[*] Loading Test data")

        rng = np.random.default_rng(seed=5566)
        Ntest = 100
        index = rng.choice(101*256, size=Ntest, replace=False)
        test_data = kappa.reshape(101*256, *shape)[index]

        self.test_data = {
            "data": test_data,
        }

    def init_submission(self, Model):
        """
        Initialize the submitted model.

        Args:
            Model (object): The model class.
        """
        print("[*] Initializing Submmited Model")

        self.model = Model()

    def fit_submission(self):
        """
        Fit the submitted model.
        """
        print("[*] Fitting Submmited Model")
        # self.model.fit(self.train_data)
        self.model.fit()

    def predict_submission(self):
        """
        Make predictions using the submitted model.
        """
        print("[*] Calling predict method of submitted model")
        self.means, self.errorbars = self.model.predict(self.test_data)

    def compute_result(self):
        """
        Compute the ingestion result.
        """
        print("[*] Computing Ingestion Result")

        def to_list(x):
            try:
                return x.tolist()
            except AttributeError:
                return x

        self.ingestion_result = {
            "means": to_list(self.means),
            "errorbars": to_list(self.errorbars)
        }

    def save_result(self, output_dir=None):
        """
        Save the ingestion result to files.

        Args:
            output_dir (str): The output directory to save the result files.
        """
        result_file = os.path.join(output_dir, "result.json")
        with open(result_file, "w") as f:
            f.write(json.dumps(self.ingestion_result, indent=4))
