from metaflow import FlowSpec, step, IncludeFile, Parameter

import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from model_data import create_train_labels
from model_symbols import train_lstm, plot_results, print_validate


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.

    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


class RK002StatsFlow(FlowSpec):
    """
    A flow to generate some statistics about the movie genres.

    The flow performs the following steps:
    1) Create training data.
    2) Create list of hyper-parameters.
    3) Compute model for each hyper-parameter.
    4) Save a dictionary of hyper-parameter specific statistics.

    """

    # movie_data = IncludeFile("movie_data",
    #                         help="The path to portfolios metadata file.",
    #                         default=script_path('interview_challenge_all.pkl')) # 'movies.csv'))
    alpha = Parameter(
        "alpha",
        help="Number of branches for hyper-parameter runs.",
        type=int,
        default=3,
    )

    @step
    def start(self):
        """
        The start step:
        1) Loads data.
        2) Define Hyperparamete NUM_SELECT_FEATURES, example: [10, 20, 30]
        3) Launches parallel statistics computation for NUM_SELECT_FEATURES genre.

        """
#        import pandas
#        from io import StringIO
        import numpy as np

        # Load the data set into a pandas dataframe.
        # Hyper-params
        # Hyperparameters
        model_name = "best_model_1_PP_001.h5"
        y_name = "y_1day_future_price_change_pct"
        # x5  # length of sequences in LSTM
        Nt_backward = 5 # 10
        # 5  # 10  # number of features for each x_t datapoint in LSTM
        Num_features = 5 # 10
        # 1-1000 training for portfolios not independently may improve prediction
        Num_portfolios = 1
        # remove high redundant correlated features
        Threshhold_cos = 0.95

        # Data
        x_train, y_label, list_portfolios = create_train_labels(
            y_name, Num_portfolios, Threshhold_cos, Num_features, Nt_backward
        )

        dim_x = x_train[0].shape[1]
        print(x_train.__len__(), x_train[0].shape)
        assert x_train[0].shape[0] == Nt_backward
        assert dim_x == Num_features + len(list_portfolios) or dim_x == Num_features

        self.x_train = x_train
        self.y_label = y_label
        self.model_name = model_name

        # self.dataframe = np.ones(int(self.alpha))   # pandas.read_csv(StringIO(self.movie_data))
        self.genres = list(
            100 * np.array(range(1, self.alpha + 1))
        )  # list([100,200])

        # We want to compute some statistics for each genre. The 'foreach'
        # keyword argument allows us to compute the statistics for each genre in
        # parallel (i.e. a fan-out).
        self.next(self.compute_statistics, foreach="genres")

    @step
    def compute_statistics(self):
        """
        Compute model for a single hyperparameter.

        """

        import numpy as np

        # The genre currently being processed is a class property called
        # 'input'.
        self.genre = self.input
        print("Computing statistics for %s" % self.genre)

        # 0.01  # regularisation
        regul_eps = 0.02
        # lstm neaurons, can be also X.shape[2]
        N1_LSTM = 10
        N2_LSTM = math.floor(N1_LSTM / 2)
        patience = 2000
        # epochs = 30  # 3000
        # validate for latest time-series points
        n_test_ratio = 0.10

        x_train = self.x_train
        y_label = self.y_label
        epochs = self.genre
        model_name = ("branch_%d_" % self.genre) + self.model_name
        (
            X_train1,
            y_train,
            X_test1,
            y_test,
            history,
            train_auc,
            test_auc,
        ) = train_lstm(
            x_train,
            y_label,
            n_test_ratio,
            regul_eps,
            epochs,
            model_name,
            N1_LSTM,
            N2_LSTM,
            patience,
        )

        # plot training history
        if False:
            plot_results(history)

            # load saved model
            saved_model = load_model(model_name)  # "best_model.h5")

            # evaluate the model
            train_auc, test_auc = print_validate(
                saved_model, X_train1, y_train, X_test1, y_test
            )

        # self.quartiles = self.genre * np.ones(3)
        self.train_auc = train_auc
        self.test_auc = test_auc
        self.history_loss = history.history["loss"]
        self.history_val_loss = history.history["val_loss"]
        self.history_auc = history.history["auc"]

        # Join the results from other genres.
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join our parallel branches and merge results into a dictionary.

        """
        # Merge results from the genre specific computations.
        if False:
            self.genre_stats = {
                inp.genre.lower(): {
                    "quartiles": inp.quartiles,
                    "dataframe": inp.dataframe,
                }
                for inp in inputs
            }
        else:
            self.genre_stats = {
                inp.genre: {
                    "train_auc": inp.train_auc,
                    "test_auc": inp.test_auc,
                    "history_auc": inp.history_auc,
                    "history_loss": inp.history_loss,
                    "history_val_loss": inp.history_val_loss,
                    "model_name": inp.model_name,
                }
                for inp in inputs
            }

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.

        """
        pass


if __name__ == "__main__":
    RK002StatsFlow()
