import pandas as pd
import numpy as np
import os
import pathlib
import plotly.express as px
from plotly.offline import plot
from pandas.api.types import CategoricalDtype
from django.conf import settings


# Default initialisation. Don't change. Instead change in first 3 lines of main() ********************************
inputDir = os.getcwd() + "/Data/Input/"
inputFile = "dataset.csv"

def createPath(filePath : str):
    fullFilePath = pathlib.Path(filePath)
    directoryPath = fullFilePath.parent
    directoryExisted = True
    if not os.path.exists(directoryPath):
        directoryExisted = False
        os.makedirs(directoryPath)
    return directoryExisted

def setInputDir(iDir : str):
    global inputDir
    if not(iDir.endswith("/")):
        iDir = iDir + "/"
    inputDir = iDir
    print("Input Directory is now", inputDir)

def setInputFile(iFile : str):
    global inputFile
    inputFile = iFile

def getDatasetFromCSV(inputDirectory : str, filename : str, printHeadAndShape : bool = False):
    if not(inputDirectory.endswith("/")):
        inputDirectory = inputDirectory + "/"
    filePath = inputDirectory + filename
    df = pd.read_csv(filePath, low_memory=False)
    if printHeadAndShape:
        print(df.shape)
        print(df.head())
        print(str(df.dtypes))
    return df
    
def makePlotly3DGraph(inputDataframe, identifierColumnName, xAxisFeatureName, yAxisFeatureName, zAxisFeatureName, sizeFeatureName, symbolFeatureName, saveFileName, xlim_lower = None, xlim_upper= None, ylim_lower = None, ylim_upper = None, zlim_lower = None, zlim_upper = None):

    inbuiltPalette = ['aqua', 'black', 'blue',
            'blueviolet', 'brown', 'cadetblue',
            'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
            'crimson', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
            'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
            'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey',
            'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
            'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
            'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
            'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
            'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
            'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
            'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
            'lightgoldenrodyellow', 'lightgray', 'lightgrey',
            'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
            'lightskyblue', 'lightslategray', 'lightslategrey',
            'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
            'linen', 'magenta', 'maroon', 'mediumaquamarine',
            'mediumblue', 'mediumorchid', 'mediumpurple',
            'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
            'mediumturquoise', 'mediumvioletred', 'midnightblue',
            'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
            'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
            'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
            'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
            'plum', 'powderblue', 'purple', 'red', 'rosybrown',
            'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon',
            'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
            'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
            'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
            'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
            'yellow', 'yellowgreen', 'blanchedalmond', 'azure',
            'beige', 'bisque', 'aliceblue', 'antiquewhite', 'cornsilk',
            'burlywood', 'aquamarine']

    fixed_symbols = ['circle', 'circle-open', 'cross', 'diamond','diamond-open', 'square', 'square-open', 'x']

    fig = px.scatter_3d(inputDataframe, x=xAxisFeatureName, y=yAxisFeatureName, z=zAxisFeatureName,
                        color=identifierColumnName, size=sizeFeatureName, size_max=3,
                        symbol=symbolFeatureName, opacity=0.7, color_discrete_sequence = inbuiltPalette,
                        symbol_sequence=fixed_symbols)

    xlim_l = xlim_lower
    xlim_u = xlim_upper
    ylim_l = ylim_lower
    ylim_u = ylim_upper
    zlim_l = zlim_lower
    zlim_u = zlim_upper
    xRange = inputDataframe[xAxisFeatureName]
    yRange = inputDataframe[yAxisFeatureName]
    zRange = inputDataframe[zAxisFeatureName]
    if xlim_lower == None:
        minValue = min(xRange)
        maxValue = max(xRange)
        xlim_l = minValue - (maxValue - minValue) * 0.075
    if ylim_lower == None:
        minValue = min(yRange)
        maxValue = max(yRange)
        ylim_l = minValue - (maxValue - minValue) * 0.075
    if zlim_lower == None:
        minValue = min(zRange)
        maxValue = max(zRange)
        zlim_l = minValue - (maxValue - minValue) * 0.075
    if xlim_upper == None:
        minValue = min(xRange)
        maxValue = max(xRange)
        xlim_u = maxValue + (maxValue - minValue) * 0.075
    if ylim_upper == None:
        minValue = min(yRange)
        maxValue = max(yRange)
        ylim_u = maxValue + (maxValue - minValue) * 0.075
    if zlim_upper == None:
        minValue = min(zRange)
        maxValue = max(zRange)
        zlim_u = maxValue + (maxValue - minValue) * 0.075

    fig.update_traces(marker=dict(size=3,
                                  line=dict(width=0,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[xlim_l, xlim_u], ),
            yaxis=dict(nticks=4, range=[ylim_l, ylim_u], ),
            zaxis=dict(nticks=4, range=[zlim_l, zlim_u], ),
            xaxis_showspikes=False,
            yaxis_showspikes=False
        ),
        scene_aspectmode='cube',
        width=1100,
        margin=dict(r=20, l=10, b=10, t=10))

    if saveFileName is not None:
        split_tup = os.path.splitext(os.path.abspath(saveFileName))
        pngFile = split_tup[0] + ".png"
        print("Generating 2-D png")
        fig.write_image(pngFile)
    else:
        try:
            # fig.show()
            plt_div = plot(fig, output_type='div',include_plotlyjs=False,show_link=False, link_text="")
            return plt_div
        except:
            print("Error encountered ")

def main():
    # Start of Input parameters ********************************
    setInputDir(f"{settings.BASE_DIR}/")
    setInputFile("clients_data.csv")
    # End of Input parameters ********************************

    # Start of feature names ********************************
    clientIDFeature = 'client_id'
    reportDateFeature = 'report_date'
    bucketRatio1Feature = 'bucketRatio1'
    bucketRatio2Feature = 'bucketRatio2'
    bucketRatio3Feature = 'bucketRatio3'
    investedAmountFeature = 'invested_amount'
    absolutePLFeature = 'total_upl'
    percentPLFeature = 'Percent Profit / Loss'
    weightedRiskFeature = 'Weighted Risk of Portfolio'
    scaledInvestmentFeature = 'Scaled Investment Amount'
    reportDateSequenceFeature = 'Date Sequence'
    clientIDNumericFeature = 'Client_id_numeric'
    # End of feature names ********************************

    originalDataframe = getDatasetFromCSV(inputDir, inputFile)

    # Start of risk multiplier value ********************************
    highRiskMultiplier = 9
    mediumRiskMultiplier = 3
    lowRiskMultiplier = 1
    # End of risk multiplier value ********************************

    # Remove -ive bucket ratios ********************************
    originalDataframe[bucketRatio1Feature] = np.where(originalDataframe[bucketRatio1Feature] < 0, 0,
                                                      originalDataframe[bucketRatio1Feature])
    originalDataframe[bucketRatio2Feature] = np.where(originalDataframe[bucketRatio2Feature] < 0, 0,
                                                      originalDataframe[bucketRatio2Feature])
    originalDataframe[bucketRatio3Feature] = np.where(originalDataframe[bucketRatio3Feature] < 0, 0,
                                                      originalDataframe[bucketRatio3Feature])

    # Create individual series for key parameters, in case they are useful later in the program ********
    investedAmount = originalDataframe[investedAmountFeature]
    absolutePL = originalDataframe[absolutePLFeature]
    bucketRatio1 = originalDataframe[bucketRatio1Feature]
    bucketRatio2 = originalDataframe[bucketRatio2Feature]
    bucketRatio3 = originalDataframe[bucketRatio3Feature]
    originalDataframe[reportDateFeature] = pd.to_datetime(originalDataframe[reportDateFeature])

    # Create a Date Sequence, for easier display (and probable sorting) ********************************
    uniqueDates = originalDataframe[reportDateFeature].unique()
    uniqueDates = np.sort(uniqueDates)
    uniqueDates = pd.DataFrame(data=uniqueDates,
                               columns=[reportDateFeature])
    dateMapping = {item: i for i, item in enumerate(uniqueDates[reportDateFeature])}
    originalDataframe[reportDateSequenceFeature] = originalDataframe[reportDateFeature].apply(lambda x: dateMapping[x])

    # Create a Client Sequence, for easier ability to allocate colours per client for Go charts ******
    uniqueClientIds = originalDataframe[clientIDFeature].unique()
    uniqueClientIds = np.sort(uniqueClientIds)
    uniqueClientIds = pd.DataFrame(data=uniqueClientIds,
                                   columns=[clientIDFeature])
    clientIDMapping = {item: i for i, item in enumerate(uniqueClientIds[clientIDFeature])}
    originalDataframe[clientIDNumericFeature] = originalDataframe[clientIDFeature].apply(lambda x: clientIDMapping[x])

    # Create %P&L and  weightedRisk features******
    percentPL = (absolutePL * 1.) / investedAmount
    weightedRisk = (lowRiskMultiplier * bucketRatio1 + mediumRiskMultiplier * bucketRatio2 + highRiskMultiplier * bucketRatio3) / (
                               (lowRiskMultiplier + mediumRiskMultiplier + highRiskMultiplier) * 100.)

    originalDataframe[percentPLFeature] = percentPL
    originalDataframe[weightedRiskFeature] = weightedRisk

    # print(originalDataframe.head(10))
    # originalDataframe.info()

    # Order clients by descending value of performance on latest date ******
    latestDate = max(uniqueDates[reportDateFeature])
    latestDatePerformance = originalDataframe[originalDataframe[reportDateFeature] == latestDate]
    latestDatePerformance = latestDatePerformance.sort_values(by=[percentPLFeature], ascending=False)
    clientIDListByPerformance = latestDatePerformance[clientIDFeature].tolist()

    # Sort dataframe, grouped by clients, with clients ordered as per above performance list ******
    client_id_order = CategoricalDtype(clientIDListByPerformance, ordered=True)
    originalDataframe[clientIDFeature] = originalDataframe[clientIDFeature].astype(client_id_order)
    originalDataframe = originalDataframe.sort_values([clientIDFeature, reportDateSequenceFeature])

    # make3DPyPlot(inputDataframe=originalDataframe, identifierColumnName=clientIDFeature, xAxisFeatureName=reportDateSequenceFeature, yAxisFeatureName=percentPLFeature, zAxisFeatureName=weightedRiskFeature, saveFileName = None, saveAsGif = False, angleStep = 3, requiredFPS = 3, figureSize = (16, 10), curveFittingOrder = 0)
    return makePlotly3DGraph(inputDataframe=originalDataframe, identifierColumnName=clientIDFeature,
                      xAxisFeatureName=reportDateSequenceFeature, yAxisFeatureName=percentPLFeature,
                      zAxisFeatureName=weightedRiskFeature, sizeFeatureName=None, symbolFeatureName=clientIDFeature,
                      saveFileName=None, xlim_lower=None, xlim_upper=None, ylim_lower=None, ylim_upper=1.,
                      zlim_lower=None, zlim_upper=None)

if __name__ == "__main__":
    main()
