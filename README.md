# GO visualizer

The GO visualizer is able to visualize the performance of 2 models that predict gene ontology terms. 
This tool makes it possible to see which model performs best on certain parts of the gene ontology tree.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and usage purposes.

### Prerequisites

* A [Linux](https://www.ubuntu.com/download/desktop) system such as Ubuntu. (Ubuntu 16.04 LTS has been used for testing.)
* A working internet connection

### Installing

1. Download the source code. This can be done by clicking the download button

2. Unzip the source code. Thist can be done by right clicking the file and clicking on "Extract Here"

3. Open the terminal with Crtl + Alt + T and set the src folder as working directory

4. Installing the dependencies can be done by running getDependencies.sh

```
cd ~/Downloads/bep_goviz-dev-f0fb4688d9a705344b9e3ce45fed4617a4402052/src
sudo bash getDependencies.sh
```

### Starting the application

The application can be started by running this command:

```
bash run.sh
```

A graphical user interface opens up and can be used to select input files.
Some example files can be found in the exampleinput directory.

The tool needs:
* An [.obo file](http://tomodachi.berkeleybop.org/amigo/landing)
* One or two files with prediction scores from models
* An importance file *Optional*

The button becomes clickable after submitting the files.
When the application is done with the visualization process, a browser window will pop up with the results.

## FAQ
**Q:** The Bokeh server starts but the interactive features do not work.

**A:** The default settings of the Tornado library do not support big web socket message sizes. This value should be increased to resolve this issue. The file is located in */usr/local/lib/python3.5/dist-packages/bokeh/server/tornado.py*.

This line:
```
super(BokehTornado, self).__init__(all_patterns)
```
Should be changed to this line:
```
 super(BokehTornado, self).__init__(all_patterns, websocket_max_message_size=50 * 1024 * 1024)
```

**Q:** The run script or one of the script it calls won't run.

**A:** Give all the called files permission to run as an application by selecting them and right clicking on properties. Go to the permissions tab and check the "Execute" checkbox.


## Dependencies
* **[Python](https://www.python.org/downloads/)** - *3.6.3*
* **[Bokeh](https://bokeh.pydata.org/en/latest/docs/installation.html)** - *0.12.16*
* **[Tornado](http://www.tornadoweb.org/en/stable/)** - *4.4.3*
* **[Colorcet](https://github.com/bokeh/colorcet)** - *1.0.0*
* **[Numpy](http://www.numpy.org/)** - *1.14.3*
* **[Pandas](https://pandas.pydata.org/getpandas.html)** - *0.22.0*
* **[Numba](https://numba.pydata.org/download.html#)** - *0.37.0*
* **[Scipy](https://www.scipy.org/install.html)** - *0.19.1*
* **[cvxopt](http://cvxopt.org/download/index.html)** - *1.1.8*
* **[Java](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)** - *1.8.0_121*
* **[JavaFX](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)** - *8.0.172*

## Authors
* **[Thomas Abeel](https://gitlab.noshit.be/thomas)** - *Supervisor*
* **[Stavros Makrodimitris](https://gitlab.noshit.be/stavros)** - *Supervisor*
* **[Jasper van Dalum](https://gitlab.noshit.be/jasper51297)** - *Contributor*
* **[Sven Bijleveld](https://gitlab.noshit.be/Sven)** - *Contributor*
* **[Bas Remmelswaal](https://gitlab.noshit.be/Basleiden)** - *Contributor*
* **[Karolis Cremers](https://gitlab.noshit.be/Karolis)** - *Contributor*

See also the list of [contributors](https://gitlab.noshit.be/AbeelLab/bep_goviz/project_members) who participated in this project.