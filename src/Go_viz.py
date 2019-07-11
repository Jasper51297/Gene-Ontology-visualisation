#!/usr/bin/python3

#Imports
import argparse
from bokeh.events import Tap
from bokeh.io import curdoc
from bokeh.layouts import (
    row,
    column
    )
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    TapTool,
    MultiLine,
    OpenURL,
    BasicTicker,
    ColorBar,
    DataTable,
    TableColumn,
    TextInput,
    Range1d,
    Toggle,
    Selection,
    VBar
    )
from bokeh.models.widgets import (
    RangeSlider,
    Button,
    PreText)
from bokeh.plotting import figure
import colorcet as cc
import numpy as np
import pandas as pd
import time
from scipy import stats
from scipy.signal import savgol_filter

def getArgs():
    """
    Retrieves the input parameters.
    --models: Amount of models, either 1 or 2
    --importance: Specified if importance is known, unspecified if not.
    """
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--models', help="Either 1 or 2 models", type=int, required=True)
    parser.add_argument('--importance', help="Only specify when importance is known", default=False, action='store_true')
    args = parser.parse_args()
    print("Selected arguments:")
    print("Amount of models:", args.models)
    print("Available importance file:", args.importance)
    return args

def createDFS():
    """
    Reads the dataframes from the pickle files.
    """
    df = pd.read_pickle("terms_dataframe.pkl")
    df = df.replace('NA', 0)
    a = df['Importance'].tolist()
    df['Score1'].fillna(0)
    if args.importance:
        df['Importance'].fillna(0)
    if args.models == 2:
        df['Score2'].fillna(0)
    link = pd.read_pickle("edges_dataframe.pkl")
    return df, link

def search_select(search_value, accuracy_value, desc_value, importance_value=(0,0)):
    """
    Arguments: Input values for search query.
    Filters rownumbers from the terms dataframe of nodes that should be selected.
    Returns the list of rownumbers.
    """
    selected = df
    if search_value != "":
        selected = selected[selected.Term.str.contains('|'.join(str(search_value).replace(', ', ',').split(',')), case=False) == True]
    else:
        selected = df
    if desc_value != "":
        selected = selected[selected.SearchDescription.str.contains('|'.join(str(desc_value).replace(', ', ',').split(',')), case=False) == True]
    else:
        selected = selected
    selected = selected[(selected.Score1 >= accuracy_value[0]) & (selected.Score1 <= accuracy_value[1])]
    if args.models == 2:
        selected = selected[(selected.Score2 >= accuracy_value[0]) & (selected.Score2 <= accuracy_value[1])]
    if args.importance:
        selected = selected[(selected.Importance >= importance_value[0]) & (selected.Importance <= importance_value[1])]
    return selected.index.tolist()

def update():
    """
    Updates the select model in the nodes dataframe.
    Visibility is triggered to update the dataframe
    """
    if args.importance:
        select = search_select(search.value, accuracy_slider.value, searchDesc.value, importance_slider.value)
    else:
        select = search_select(search.value, accuracy_slider.value, searchDesc.value)
    source.selected.indices = select
    selectedTerm.value=', '.join(df['Term'].iloc[select].tolist())
    if nodes1.visible:
        nodes1.visible = False
        nodes1.visible = True
    else:
        nodes1.visible = True
        nodes1.visible = False
    if node4.visible and args.importance:
        node4.visible = False
        node4.visible = True
    else:
        node4.visible = True
        node4.visible = False
    
def showDiff(active):
    """
    Hides/shows the scores from model 1 and 2 and shows the difference between them
    """
    global Active
    if active:
        edges_2.visible = True
        nodes3.visible = True
        edges_r.visible = False
        nodes2.visible = False
        nodes1.visible = False
        toggleDiff.label = 'Show models'
        Active = True
    else:
        edges_2.visible = False
        nodes3.visible = False
        edges_r.visible = True
        nodes2.visible = True
        nodes1.visible = True
        toggleDiff.label = 'Show differences'
        Active = False

def getAncestors(goid):
    """
    Argument: Selected node
    Recursive function to retrieve the ancestors of the selected node.
    Returns a list with the rownumbers of the ancestors nodes.
    """
    recursiveArray = [goid]
    parents = source.data['Parents'][goid]
    if len(parents) > 0:
        for parent in parents:
            global linklst
            linklst.append([goid, parent])
            recursiveArray.extend(getAncestors(parent))
    return recursiveArray

def getDescendents(goid):
    """
    Argument: Selected node
    Recursive function to retrieve the children of the selected node.
    Returns a list with the rownumbers of the children nodes.
    """
    recursiveArray = [goid]
    children = source.data['Children'][goid]
    if len(children) > 0:
        for child in children:
            global linklst
            linklst.append([goid, child])
            recursiveArray.extend(getDescendents(child))
    return recursiveArray

def tapped(attr, old, new):
    """
    Arguments: Must be there, are not used.
    Removes lines between nodes and clears the textfields when no nodes
    are selected.
    Creates lines between the selected nodes if 1 node is selected. Adds
    selected node to the selected textfield and the children/parents to
    the children/parents field.
    Removes lines between the nodes if multiple nodes are selected and
    adds them to the selected textfield.
    """
    global start_time
    elapsed_time = time.time()-start_time
    if elapsed_time > 0.5:
        if args.models == 2:
            if Active:
                edges_2.visible = True
            else:
                edges_r.visible = True
        else:
            edges_r.visible = True
        sr.visible = False
        sn.visible = False
        if len(source.selected.indices) == 0:
            selectedTerm.value=''
            parentlist.value=''
            childrenlist.value=''
            if args.models == 2:
                if Active:
                    edges_2.visible = True
                else:
                    edges_r.visible = True
            else:
                edges_r.visible = True
            sr.visible = False
            sn.visible = False
        elif len(source.selected.indices) == 1:
            global linklst
            linklst = []
            selectedData = {'x': [df.loc[source.selected.indices[0]]['X']], 'top': [df.loc[source.selected.indices[0]]['Top']], 'bottom': [df.loc[source.selected.indices[0]]['Bottom']]}
            showSelected.data = selectedData
            parents = getAncestors(source.selected.indices[0])
            children = getDescendents(source.selected.indices[0])
            start_time = time.time()
            select = parents + children
            source.selected = Selection(indices=select)
            linkdata = {'x0': [], 'y0': [], 'x1': [], 'y1': []};
            for i in range(len(linklst)):
                linkdata['x0'].append(source.data.get('X')[linklst[i][0]])
                linkdata['y0'].append(source.data.get('Y')[linklst[i][0]])
                linkdata['x1'].append(source.data.get('X')[linklst[i][1]])
                linkdata['y1'].append(source.data.get('Y')[linklst[i][1]])
            showlinks.data = linkdata
            if args.models == 2:
                if Active:
                    edges_2.visible = False
                else:
                    edges_r.visible = False
            else:
                edges_r.visible = False
            sr.visible = True
            sn.visible = True
            p, c = [], []
            for i in parents:
                p.append(source.data['Term'][i])
            for i in children:
                c.append(source.data['Term'][i])
            del c[0]
            selectedTerm.value=p.pop(0)
            parentlist.value=', '.join(p)
            childrenlist.value=', '.join(c)
            return
        elif len(source.selected.indices) > 1:
            selectedTerm.value=', '.join(df['Term'].iloc[source.selected.indices].tolist())
            parentlist.value=''
            childrenlist.value=''
            if args.models == 2:
                if active:
                    edges_2.visible = True
                else:
                    edges_r.visible = True
            else:
                edges_r.visible = True
            sr.visible = False
            sn.visible = False
        start_time = time.time()

def outliers(group):
    """
    Finds outliers per layer 
    """
    layer = group.name
    if args.models == 2:
        return group[(group.Difference > upper.loc[layer]['Difference']) | (group.Difference < lower.loc[layer]['Difference'])]['Difference']
    else:
        return group[(group.Score1 > upper.loc[layer]['Score1']) | (group.Score1 < lower.loc[layer]['Score1'])]['Score1']

start_time = time.time()
Active = False
linklst = []
args = getArgs()
df, link = createDFS()
#Add the fields, buttons and sliders for the search function
search = TextInput(title="Search GO term")
searchDesc = TextInput(title="Search description")
accuracy_slider = RangeSlider(start=0, end=1, value=(0,1), step=.01, title="Accuracy")
if args.importance:
    importance_slider = RangeSlider(start=float(df['Importance'].min()), end=float(df['Importance'].max()), value=(float(df['Importance'].min()), float(df['Importance'].max())), step=.01, title="Importance")
button = Button(label="Search", button_type="success")
button.on_click(update)
if args.models == 2:
    toggleDiff = Toggle(label="Show differences", button_type="success")
    toggleDiff.on_click(showDiff)
#Creates color mappers/legends
color_mapper = LinearColorMapper(palette=cc.gray[::-1], low=0, high=1)
color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title='Accuracy',
                     label_standoff=12, location=(0,0))
#Set plot properties
tools = "pan,box_zoom,reset,xwheel_zoom,save"
plot = figure(title="GO visualizer", sizing_mode='stretch_both', tools = tools, output_backend="webgl", width=1400, height=840, toolbar_location="above", active_scroll="xwheel_zoom")
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None
plot.axis.visible = False
toolStat1 = "pan,box_zoom,wheel_zoom,reset,save"
plotStat1 = figure(title="Information content VS performance", sizing_mode='stretch_both', tools = toolStat1, output_backend="webgl", width=650, height=650, toolbar_location="above")
plotStat1.xaxis.axis_label = "Information content"
plotStat1.yaxis.axis_label = "Performance"
#Convert dataframe to datasource
linksource = ColumnDataSource(link)
source = ColumnDataSource(df)
#Draw nodes and lines for go visualiser and statistics plots
lines = MultiLine(xs='x', ys='y', line_alpha=0.2, name='edge', line_width=2.5, line_color='darksalmon')
edges_r = plot.add_glyph(source_or_glyph=linksource, glyph=lines)
if args.models == 2:
    lines2 = MultiLine(xs='x', ys='y', line_alpha=0.1, name='edge', line_width=2.5, line_color='grey')
    edges_2 = plot.add_glyph(source_or_glyph=linksource, glyph=lines2)
    edges_2.visible = False
    node1 = VBar(x="X", top="Top", bottom="Y", width=0.95, fill_color={'field': 'Score1', 'transform': color_mapper}, name='node1', line_width=0.1, line_color='black', fill_alpha=1)
    node1_unselected = VBar(x="X", top="Top", bottom="Y", width=0.95, fill_color={'field': 'Score1', 'transform': color_mapper}, name='node1', line_width=0.1, line_color='black', fill_alpha=0.05)
    nodes1 = plot.add_glyph(source_or_glyph=source, glyph=node1, nonselection_glyph=node1_unselected)
    node2 = VBar(x="X", top="Y", bottom="Bottom", width=0.95, fill_color={'field': 'Score2', 'transform': color_mapper}, name='node2', line_width=0.1, line_color='black', fill_alpha=1)
    node2_unselected = VBar(x="X", top="Y", bottom="Bottom", width=0.95, fill_color={'field': 'Score2', 'transform': color_mapper}, name='node2', line_width=0.1, line_color='black', fill_alpha=0.05)
    nodes2 =plot.add_glyph(source_or_glyph=source, glyph = node2, nonselection_glyph=node2_unselected)
    color_mapper_diff = LinearColorMapper(palette=cc.coolwarm, low=df['Difference'].min(), high=df['Difference'].max())
    color_bar_diff = ColorBar(color_mapper=color_mapper_diff, ticker=BasicTicker(), title='Difference',
                     label_standoff=12, location=(0,0))
    node3 = VBar(x="X", top="Top", bottom="Bottom", width=0.95, fill_color={'field': 'Difference', 'transform': color_mapper_diff}, name='node3', line_width=0.1, line_color='black', fill_alpha=1)
    node3_unselected = VBar(x="X", top="Top", bottom="Bottom", width=0.95, fill_color={'field': 'Difference', 'transform': color_mapper_diff}, name='node3', line_width=0.1, line_color='black', fill_alpha=0.05)
    nodes3 = plot.add_glyph(source_or_glyph=source, glyph = node3, nonselection_glyph= node3_unselected)
    nodes3.visible = False
    taptool = TapTool(renderers=[nodes1, nodes2])
else:
    node1 = VBar(x="X", top="Top", bottom="Bottom", width=0.95, fill_color={'field': 'Score1', 'transform': color_mapper}, name='node1', line_width=0.1, line_color='black', fill_alpha=1)
    node1_unselected = VBar(x="X", top="Top", bottom="Bottom", width=0.95, fill_color={'field': 'Score1', 'transform': color_mapper}, name='node1', line_width=0.1, line_color='black', fill_alpha=0.05)
    nodes1 = plot.add_glyph(source_or_glyph=source, glyph = node1, nonselection_glyph = node1_unselected)
    taptool = TapTool(renderers=[nodes1])
if args.importance:
    node4 = plotStat1.circle(x="Importance", y="Score1", source=source, color="navy", size=1, legend = "Model 1")
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['Importance'], df['Score1'])
    rhoS, pvalS = stats.spearmanr(df['Importance'], df['Score1'])
    pvalP = stats.pearsonr(df['Score1'], df['Importance'])
    text = "Model 1:\nP value Pearson correlation: " + str(pvalP[1]) + "\nP value Spearman: " + str(pvalS) + "\n"
    x = df['Importance'].tolist()
    y = df['Score1'].tolist()
    len1 = len(x)
    x, y = zip(*sorted(zip(x, y)))
    x = list(filter(lambda a: a != 0, x))
    len2 = len(x)
    y = y[len1-len2::]
    window = int(len(x)/3)
    if window % 2 == 0:
        window += 1
    ysmooth = savgol_filter(y, window, 2)
    plotStat1.line(x=x, y=ysmooth, legend = "Model 1", line_color="navy")
    if args.models == 2:
        plotStat1.circle(x="Importance", y="Score2", source=source, color="red", size=1, legend = "Model 2")
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Importance'], df["Score2"])
        x = df['Importance'].tolist()
        y = df['Score2'].tolist()
        len1 = len(x)
        x, y = zip(*sorted(zip(x, y)))
        x = list(filter(lambda a: a != 0, x))
        len2 = len(x)
        y = y[len1-len2::]
        window = int(len(x)/3)
        if window % 2 == 0:
            window += 1
        ysmooth = savgol_filter(y, window, 1)
        plotStat1.line(x=x, y=ysmooth, legend = "Model 2", line_color="red")
        rhoS, pvalS = stats.spearmanr(df['Importance'], df['Score2'])
        pvalP = stats.pearsonr(df['Importance'], df['Score2'])
        text2 =  "Model 2:\nP value Pearson correlation: " + str(pvalP[1]) + "\nP value Spearman: " + str(pvalS) + "\n"
        plotStat1.circle(x="Importance", y="Difference", source=source, color="green", size=1, legend = "Differences")
        x = df['Importance'].tolist()
        y = df['Difference'].tolist()
        len1 = len(x)
        x, y = zip(*sorted(zip(x, y)))
        x = list(filter(lambda a: a != 0, x))
        len2 = len(x)
        y = y[len1-len2::]
        x, y = zip(*sorted(zip(x, y)))
        window = int(len(x)/3)
        if window % 2 == 0:
            window += 1
        ysmooth = savgol_filter(y, window, 1)
        plotStat1.line(x=x, y=ysmooth, legend = "Differences", line_color="Green")
        rhoS, pvalS = stats.spearmanr(df['Importance'], df['Difference'])
        pvalP = stats.pearsonr(df['Importance'], df['Difference'])
        text3 =  "Differences:\nP value Pearson correlation: " + str(pvalP[1]) + "\nP value Spearman: " + str(pvalS) + "\n"
        text = str(text) + str(text2) + str(text3)
    plotStat1.legend.location = "top_left"
    plotStat1.legend.click_policy="hide"
    statText1 = PreText(text=text, width=700)
#Create a hover tool for the edges and nodes
if int(args.models) == 2:
    tooltips = [('Label', "@Term"), ('Score model 1', '@Score1'), ('Score model 2', '@Score2'), ('Difference', '@Difference'), ('Namespace', '@Namespace'), ('Description', '@Description')]
    if args.importance:
        tooltips = [('Label', "@Term"), ('Score model 1', '@Score1'), ('Score model 2', '@Score2'), ('Difference', '@Difference'), ('Importance', '@Importance'), ('Namespace', '@Namespace'), ('Description', '@Description')]
    hoverNodes = HoverTool(renderers=[nodes1, nodes2, nodes3],
                      tooltips=tooltips)
else:
    tooltips = [('Label', "@Term"), ('Score model', '@Score1'), ('Namespace', '@Namespace'), ('Description', '@Description')]
    if args.importance:
        tooltips = [('Label', "@Term"), ('Score model', '@Score1'), ('Importance', '@Importance'), ('Namespace', '@Namespace'), ('Description', '@Description')]
    hoverNodes = HoverTool(renderers=[nodes1],
                      tooltips=tooltips)
plot.add_tools(hoverNodes)
hoverEdges = HoverTool(renderers=[edges_r],
                  tooltips=[('Parent', "@Term1"),
                            ('Child', "@Term2"),
                            ('Type', '@Type')])
plot.add_tools(hoverEdges)
#Create a taptool that opens the gene ontology website with the clicked GO term
url = "http://amigo2.berkeleybop.org/amigo/term/@Term/"
taptool.callback = OpenURL(url=url)
plot.add_tools(taptool)
#Show parents and children on select in the visualiser and in the text fields
showlinks = ColumnDataSource({'x0': [], 'y0': [], 'x1': [], 'y1': []})
showSelected = ColumnDataSource({'x': [], 'top': [], 'bottom': []})
sr = plot.segment(x0='x0', y0='y0', x1='x1', y1='y1', color='olive', alpha=0.6, line_width=2, source=showlinks)
sn = plot.vbar(x ="x", top='top', bottom='bottom', width=1, color='red', alpha=0.5, source=showSelected)
parentlist = TextInput(value="", title="Parents:")
childrenlist = TextInput(value="", title="Children:")
selectedTerm = TextInput(value="", title="Selected term:")
ancestryTool = TapTool()
plot.add_tools(ancestryTool)
#Table
if args.models == 2:
    columns = [
        TableColumn(field="Term", title="Term"),
        TableColumn(field="Score1", title="Score 1"),
        TableColumn(field="Score2", title="Score 2"),
        TableColumn(field="Difference", title="Difference")
        ]
    if args.importance:
        columns = [
            TableColumn(field="Term", title="Term"),
            TableColumn(field="Score1", title="Score 1"),
            TableColumn(field="Score2", title="Score 2"),
            TableColumn(field="Difference", title="Difference"),
            TableColumn(field="Importance", title="Importance")
            ]
else:
    columns = [
        TableColumn(field="Term", title="Term"),
        TableColumn(field="Score1", title="Score 1")
        ]
    if args.importance:
        columns = [
            TableColumn(field="Term", title="Term"),
            TableColumn(field="Score1", title="Score 1"),
            TableColumn(field="Importance", title="Importance")
            ]

#Call function tapped when something is changed in the node source
source.on_change('selected', tapped)
data_table = DataTable(source=source, columns=columns, width=300, height=560)

#Create boxplot with performance difference vs layer
dfbox = df.loc[df['Score1'] != 0]
dfbox['Layer'] = df['Layer'].astype(int)
groups = dfbox.groupby('Layer')
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr
out = groups.apply(outliers).dropna()
layers = list(groups.groups.keys())
layers.sort(key=int)
if not out.empty:
    outx = []
    outy = []
    for layer in layers:
        if not out.loc[layer].empty:
            for value in out[layer]:
                outx.append(layer)
                outy.append(value)
toolStat2 = "wheel_zoom, save"
plotStat2 = figure(title="GO layer VS performance", sizing_mode='stretch_both', tools = toolStat2, width=650, height=650, toolbar_location="above")
plotStat2.xaxis.ticker = layers
plotStat2.xaxis.axis_label = "GO layer"
if args.models == 2:
    plotStat2.yaxis.axis_label = "Difference performance"
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.Difference = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'Difference']),upper.Difference)]
    lower.Difference = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'Difference']),lower.Difference)]
    plotStat2.segment(layers, upper.Difference, layers, q3.Difference, line_color="black")
    plotStat2.segment(layers, lower.Difference, layers, q1.Difference, line_color="black")
    plotStat2.vbar(x=layers, width=0.7, bottom=q2.Difference, top=q3.Difference, fill_color="darksalmon", line_color="black")
    plotStat2.vbar(x=layers, width=0.7, bottom=q1.Difference, top=q2.Difference, fill_color="darksalmon", line_color="black")
    plotStat2.rect(layers, lower.Difference, 0.2, 0.001, line_color="black")
    plotStat2.rect(layers, upper.Difference, 0.2, 0.001, line_color="black")
    if not out.empty:
        plotStat2.circle(outx, outy, size=6, color="darksalmon", fill_alpha=0.6)
else:
    plotStat2.yaxis.axis_label = "Performance"
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.Score1 = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'Score1']),upper.Score1)]
    lower.Score1 = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'Score1']),lower.Score1)]
    plotStat2.segment(layers, upper.Score1, layers, q3.Score1, line_color="black")
    plotStat2.segment(layers, lower.Score1, layers, q1.Score1, line_color="black")
    plotStat2.vbar(layers, 0.7, q2.Score1, q3.Score1, fill_color="darksalmon", line_color="black")
    plotStat2.vbar(layers, 0.7, q1.Score1, q2.Score1, fill_color="darksalmon", line_color="black")
    plotStat2.rect(layers, lower.Score1, 0.2, 0.001, line_color="black")
    plotStat2.rect(layers, upper.Score1, 0.2, 0.001, line_color="black")
    if not out.empty:
        plotStat2.circle(outx, outy, size=6, color="darksalmon", fill_alpha=0.6)

plot.add_layout(color_bar, 'left')
plot.toolbar.active_inspect = hoverNodes
#Add plots to the output screen
child = [data_table, search, searchDesc, accuracy_slider, button]
statPlots = []
if args.importance:
    child = [data_table, search, searchDesc, accuracy_slider, importance_slider, button]
    statPlots = [plotStat1, statText1]
if args.models == 2:
    child.append(toggleDiff)
    plot.add_layout(color_bar_diff, 'left')
curdoc().add_root(row(children=[column(children=[plot, row(children=[selectedTerm, parentlist, childrenlist]), row(children=[column(children=statPlots), plotStat2])]), column(children=child)]))
curdoc().title = "Gene Ontology Visualizer"