**Project Name**

Setup and Installation Guide

**Prerequisites**
- Python 3.8+ (currently using Python 3.11.6)
- pip (currently using pip 23.2.1)
- virtualenv (recommended)
  
**Step-by-Step Installation**
1. Create and Activate Virtual Environment

On Windows:
```
python -m venv venv
venv\Scripts\activate
```

On MacOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Rename Files for Jupyter Notebook Importing
To make importing files in Jupyter Notebook easier, renamed the following files:

- CSCN_BaselineData_MSS_2024-11-07.xlsx   --------->   features_data.xlsx
- UBC_Calgary_Ottawa_Laval_Final_Summerized_Patients_Table_Metrics_For Dr. Ayas.xlsx     --------->    patients_data.xlsx


