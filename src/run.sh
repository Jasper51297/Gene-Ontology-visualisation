#!/bin/bash
chmod +x jdk1.8.0_172/bin/java
chmod +x parse.py
chmod +x Go_viz.py
./jdk1.8.0_172/bin/java -jar goviz_GUI.jar
args="$(cat runargs.txt)"
echo ${args}
python3 parse.py ${args}
args="$(cat bokehargs.txt)"
echo ${args}
bokeh serve --show Go_viz.py --args ${args}