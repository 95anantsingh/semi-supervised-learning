import os
import torch
import warnings
import pandas as pd
import seaborn as sns
import plotly.express as px
from dash import Dash, html, dcc


warnings.filterwarnings("ignore")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
if sns.__version__ != '0.12.1': raise NotImplementedError

if __name__ =='__main__':
    # Load history files
    history_dir = './DBpedia/history_pc_top_k/'
    history_files = []
    history_list = []
    for root, dirs, files in os.walk(history_dir):
        for file in files:
            if file.split('.')[-1]=='pt':
                history_files.append(file)
    history_list = [torch.load(history_dir+history_file) for history_file in (history_files)]


    # Make data
    data = {'Iteration':[], 'Accuracy':[], 'Per Class K':[], 'Labeled Size':[], 'Unlabeled Size':[], 'Unlabeled Classes':[]}
    for history in history_list:
        data['Iteration']+=[*range(len(history['val_acc']))]
        data['Accuracy']+=[max(accs) for accs in  history['val_acc']]
        data['Per Class K']+= ([history['config'].pc_top_k]*len(history['val_acc']))
        data['Labeled Size']+=([history['labeled_set_size'][0]]*len(history['val_acc']))
        data['Unlabeled Size']+=([history['unlabeled_set_size'][0]]*len(history['val_acc']))
        data['Unlabeled Classes']+=([len(history['config'].unlabeled_classes)]*len(history['val_acc']))


    # Make Dataframe
    df= pd.DataFrame()
    for key, val in data.items():
        df[key]=val


    # Plot Accuracies
    color_col = 'Labeled Size'
    dash_col = 'Unlabeled Classes'
    symbol_col = 'Per Class K'

    fig = px.line(df ,x='Iteration',y='Accuracy',
                color=color_col, line_dash=dash_col, symbol=symbol_col,
                height=700, width=1400, markers=True, template="plotly_white",
                color_discrete_sequence=sns.color_palette("tab10",len(df[color_col].unique())).as_hex(),
                line_dash_sequence=['solid', 'dash'],
                symbol_sequence= ['circle', 'arrow-up', 'circle-open-dot', 'square'],
                title='Accuracy Comparison')

    # Start Flask Server
    app = Dash(__name__)
    app.layout = html.Div(children=[
        dcc.Graph(
            id='acc-comp',
            figure=fig
        )
    ])

    app.run_server(debug=True, port=8001, use_reloader=False)
    print()