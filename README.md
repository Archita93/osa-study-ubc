# Project Name

## Setup and Installation Guide

### Prerequisites

- Python 3.8+ (currently using Python 3.11.6)
- pip (currently using pip 23.2.1)
- virtualenv (recommended)

### Step-by-Step Installation

#### 1. Create and Activate Virtual Environment

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On MacOS/Linux:**

```bash
python -m venv venv
venv\Scripts\activate
```


#### 2. Install requirements
```bash
pip install -r requirements.txt
```

#### 3. Rename Files for Jupyter Notebook Importing

To make importing files in Jupyter Notebook easier, renamed the following files:
| Original Filename | New Filename |
|-------------------|--------------|
| CSCN_BaselineData_MSS_2024-11-07.xlsx | features_data.xlsx |
| UBC_Calgary_Ottawa_Laval_Final_Summerized_Patients_Table_Metrics_For Dr. Ayas.xlsx | patients_data.xlsx |

#### 4. Project Structure
```bash
project_root/
│
├── venv/
├── features_data.xlsx
├── patients_data.xlsx
├── requirements.txt
└── README.md
```

#### 5. Additional Notes
- Always activate your virtual environment before working on the project
- Ensure all dependencies are listed in requirements.txt
- If you encounter any issues, please check the versions of Python and pip

