import os
import sys
import logging
import pandas as pd
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table, TableStyle
from reportlab.lib.colors import lavender, red, green

import diagnostics
from config import DATA_PATH, MODEL_PATH, TEST_DATA_PATH
# from pretty_confusion_matrix import pp_matrix_from_data#plot_confusion_matrix_from_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def plot_confusion_matrix():
    """
    Calculate a confusion matrix using the test data and the deployed model
    plot the confusion matrix using pretty confusion matrix to the workspace
    """
    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(TEST_DATA_PATH, 'testdata.csv'))

    y_true = test_df.pop('exited')
    # print(y_true)
    
    X_df = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting test data")
    y_pred = diagnostics.model_predictions(X_df)
    # print(y_pred)
    # logging.info("Plotting and saving confusion matrix")
    # fig, ax = pp_matrix_from_data(
    #     y_true, y_pred, columns=[0, 1], cmap='Blues')
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Model Confusion Matrix')
    plt.savefig(os.path.join(MODEL_PATH, 'confusionmatrix.png'))

    # ax.set_title("Model Confusion Matrix")
    # fig.savefig(os.path.join(MODEL_PATH, 'confusionmatrix.png'))


def _get_statistics_df():
    """
    Get data statistics and missing percentage of each column
    in pandas dataframe to draw table in the PDF report

    Returns:
        pd.DataFrame: Train data summary
    """
    stats = diagnostics.dataframe_summary()
    missing = diagnostics.missing_percentage()

    data = {'Column Name': [k for k in missing.keys()]}
    data['Missing %'] = [missing[column]['percentage']
                         for column in data['Column Name']]

    temp_col = list(stats.keys())[0]
    for stat in stats[temp_col].keys():
        data[stat] = [
            round(
                stats[column][stat],
                2) if stats.get(
                column,
                None) else '-' for column in data['Column Name']]

    return data


def generate_pdf_report():
    """
    Generate PDF report that includes ingested data information, model scores
    on test data and diagnostics of execution times and packages
    """
    pdf = canvas.Canvas(
        os.path.join(
            MODEL_PATH,
            'summary_report.pdf'),
        pagesize=A4)

    pdf.setTitle("Model Summary Report")

    pdf.setFontSize(24)
    pdf.setFillColorRGB(31 / 256, 56 / 256, 100 / 256)
    pdf.drawCentredString(300, 800, "Model Summary Report")

    # Ingest data section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 750, "Ingested Data")

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 725, "List of files used:")

    # Ingested files
    with open(os.path.join(DATA_PATH, "ingestedfiles.txt")) as file:
        pdf.setFontSize(12)
        text = pdf.beginText(40, 705)
        text.setFillColor('black')

        for line in file.readlines():
            text.textLine(line.strip('\n'))

        pdf.drawText(text)

    # Data statistics and missing percentage
    data = _get_statistics_df()
    data_df = pd.DataFrame(data)
    data_table = data_df.values.tolist()
    data_table.insert(0, list(data_df.columns))

    # Draw summary table
    stats_table = Table(data_table)
    stats_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 1, 'black'),
        ('BACKGROUND', (0, 0), (-1, 0), lavender)
    ])

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 645, "Statistics Summary")

    stats_table.wrapOn(pdf, 40, 520)
    stats_table.drawOn(pdf, 40, 520)

    # Trained model section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 490, "Trained Model Scoring on Test Data")

    pdf.setFontSize(12)
    pdf.setFillColorRGB(128 / 256, 128 / 256, 128 / 256)
    pdf.drawString(25, 480, "testdata.csv")

    # Model score
    with open(os.path.join(MODEL_PATH, "latestscore.txt")) as file:
        pdf.setFontSize(12)
        pdf.setFillColor('black')
        pdf.drawString(40, 460, file.read())

    # Model confusion matrix
    pdf.drawInlineImage(
        os.path.join(
            MODEL_PATH,
            'confusionmatrix.png'),
        40,
        150,
        width=300,
        height=300)

    # New page
    pdf.showPage()

    # Diagnostics section
    pdf.setFontSize(18)
    pdf.setFillColorRGB(47 / 256, 84 / 256, 150 / 256)
    pdf.drawString(25, 780, "Diagnostics")

    # Execution time
    timings = diagnostics.execution_time()

    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 755, "Execution times:")

    pdf.setFontSize(12)
    text = pdf.beginText(40, 735)
    text.setFillColor('black')

    for time in timings:
        for k, v in time.items():
            text.textLine(f"{k} = {round(v, 4)}")

    pdf.drawText(text)

    # Draw outdated dependencies table
    data = diagnostics.outdated_packages_list()
    
    pdf.setFontSize(14)
    pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    pdf.drawString(35, 690, "Outdated Dependencies")
    
    if not data:  # If the data list is empty
        pdf.setFontSize(12)
        pdf.setFillColor('black')
        pdf.drawString(40, 675, "No outdated libraries found.")
        
    else:
        table_style = TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, 'black'),
            ('BACKGROUND', (0, 0), (-1, 0), lavender)
        ])

        for row, values in enumerate(data[1:], start=1):
            if values[1] != values[2]:
                table_style.add('TEXTCOLOR', (1, row), (1, row), red)
                table_style.add('TEXTCOLOR', (2, row), (2, row), green)

        depend_table = Table(data)
        depend_table.setStyle(table_style)

        depend_table.wrapOn(pdf, 40, 325)
        depend_table.drawOn(pdf, 40, 325)


    # table_style = TableStyle()
    # table_style.add('GRID', (0, 0), (-1, -1), 1, 'black')
    # table_style.add('BACKGROUND', (0, 0), (-1, 0), lavender)

    # for row, values in enumerate(data[1:], start=1):
    #     if(values[1] != values[2]):
    #         table_style.add('TEXTCOLOR', (1, row), (1, row), red)
    #         table_style.add('TEXTCOLOR', (2, row), (2, row), green)

    # depend_table = Table(data)
    # depend_table.setStyle(table_style)

    # pdf.setFontSize(14)
    # pdf.setFillColorRGB(46 / 256, 116 / 256, 181 / 256)
    # pdf.drawString(35, 690, "Outdated Dependencies")

    # depend_table.wrapOn(pdf, 40, 325)
    # depend_table.drawOn(pdf, 40, 325)

    pdf.save()


if __name__ == '__main__':
    logging.info("Running reporting.py")

    logging.info("Generating confusion matrix")
    plot_confusion_matrix()

    logging.info("Generating PDF report")
    generate_pdf_report()