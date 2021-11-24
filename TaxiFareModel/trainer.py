from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data, define_y, define_X
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.model_selection import train_test_split

estimat = LinearRegression()
MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[FR] [LILLE] [Rvanelven] LinearRegression v1"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.experiment_name = EXPERIMENT_NAME
        self.pipe = None
        self.X = X
        self.y = y

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        return self

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        self.pipe = Pipeline([('preproc', preproc_pipe),
                              ('linear_model', estimat)])
        return self.pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(self.X_train, self.y_train)
        return self.pipe

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:

            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:

            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        pass


if __name__ == "__main__":
    df = get_data()
    clean_data(df)
    trainer = Trainer(define_X(df), define_y(df))
    trainer.split_data()
    trainer.set_pipeline()
    trainer.run()
    trainer.evaluate()
    score = trainer.evaluate()
    trainer.mlflow_log_metric('rmse', score)
    trainer.mlflow_log_param('estimator', estimat)
