# airfoils2018
Data Repository for Airfoil DataFrames  from University of Illinois at Urbana-Champaign Airfoil Data Site, and Python scripts for the research article titled

Deep learning used to classify structural features and predict performance parameters of airfoils
Mecit Yaman myaman@thk.edu.tr
Department of Aeronautical Engineering, University of the Turkish Aeronautical Association, Ankara, Turkey

Data cleaning, regularization and pandas DataFrame preparation
 
•	dataset generator.py
•	foil.py
•	rawbase.pkl
•	coord_seligFmt.zip

Structure determination and labelling

•	1 calculate trailing edge structure.py 
•	rawbase.pkl
•	rawbase_with_structure_info.pkl

Performance parameter (lift coefficient) determination and labelling

•	foil.py
•	flow_functions.py
•	1 calculate trailing edge structure.py 
•	rawbase_with_structure_and_lift.pkl

Structure prediction using TensorFlow DNNClassifier and DNNRegressor

•	DNN_structure.py
•	structure_analytics.pkl

Lift regression using TensorFlow DNNClassifier and DNNRegressor

•	DNN_performance.py
•	performance_analytics.pkl


