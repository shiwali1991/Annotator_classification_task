# Annotator_classification_task
# About
- A quick tool to classify annotators into "Good" and "Bad" annotators. The tool relies upon three factors accuracy, level of agreement with other annotators and time. It uses the concept of weighted average to come up to a score and then classifies the annotators using kmeans.
 
# Files Description
- "Quality_Match_bicycle_task.ipynb" is the python notebook. It contains all the functions with comments as well as visualizations.
- "annotator_classify.py" is streamlit file that can be run to test the classification.
- "Quality Match_bicycle_task.ppt.pdf" is presentation pdf that contains summarized findings and future scope of work.
- "Quality_Match_bicycle_task.py" is the python file to run classification pipeline.

# how to use
- Add two data files sample annotation "anonymized_project.json" and reference data "references.json" in the same root directory. 
- Pipeline Run simply by python Quality_Match_bicycle_task.py
- Streamlit app run simply by streamlit run annotator_classify.py
- The presentation can be accessed from here as well as from the link.

# Future Work
- To work on the metric to calculate classification score.
- To work on the prototype.
- To have feedback and learning mechanism for annotators.
