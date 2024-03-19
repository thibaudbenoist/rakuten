from sklearn.metrics import classification_report
import pandas as pd
import os
import matplotlib.pyplot as plt

class Benchmark():
    """
    Class to store and display classification results
    """
    def __init__(
            self,
            file_path
            ):
        """
        Parameters:
        file_path (str): Path to the CSV file where the results will be stored.

        Returns:
        Benchmark: Object to store and display classification results.
        """
        self.file_path = file_path
        if os.path.isfile(file_path):
            self.report = pd.read_csv(file_path, index_col=0)
        else:
            self.report = None

    def add_results(
            self,
            label,
            classifier_name,
            parameters,
            y_test,
            y_pred
    ):
        """
        Add classification results to the report.

        Parameters:
        label (str): Name of the label.
        classifier_name (str): Name of the classifier.
        parameters (str): Parameters used for the classifier.
        y_test (Series): True labels.
        y_pred (Series): Predicted labels.

        Returns:
        Benchmark: Object to store and display classification results.
        """
        report = classification_report(y_test, y_pred, output_dict=True)
        labels = ['Label', 'Classifier', 'Parameters']+list(report.keys())


        values = [label, classifier_name, parameters]
        for key, value in report.items():
            if isinstance(value, dict) and 'f1-score' in value:
                values.append(round(value['f1-score'],3))
            elif key=='accuracy':
                values.append(round(value,3))
            else:
                values.append('')

        if self.report is None:
            self.report = pd.DataFrame(values, index=labels)
        else:
            self.report = pd.concat([self.report, pd.DataFrame(values, index=labels)], axis=1)

        return self 

    def save(self):
        """
        Save the report to a CSV file.
        
        Returns:
        Benchmark: Object to store and display classification results.
        """
        self.report.to_csv(self.file_path, index_label=0)

        return self

    def show_report(self, y_test):
        """
        Display the scores by product type.

        Parameters:
        y_test (Series): Number of items by product type displayed on the x-axis to compare with the scores

        Returns:
        None
        """
        ref = y_test.value_counts()
        ref.index = ref.index.astype(str)

        toplot = self.report
        toplot.columns = toplot.iloc[0]
        toplot = toplot.iloc[3:].astype(float)
        toplot = toplot.join(ref)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        toplot.drop('count', axis=1).plot(kind='bar', ax=ax1, width=0.8, rot=45)
        ax2 = ax1.twinx()
        toplot['count'].plot(kind='line',ax=ax2)
        ax2.fill_between(toplot.index, toplot['count'], color='lightgrey', alpha=0.5)
        plt.show()
