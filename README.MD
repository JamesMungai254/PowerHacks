# TB Analysis and Diagnosis

This repository contains two main components:

1. **TB_ANALYSIS.ipynb**: A Jupyter Notebook that performs analysis of TB chest X-ray images and model creation using MONAI.
2. **tbcxrdiagnosis**: A Django application that deploys the model and provides a user interface for medical personnel to analyze chest X-ray images.

## MONAI

MONAI (Medical Open Network for AI) is a PyTorch-based framework designed specifically for deep learning in healthcare imaging. It provides domain-optimized foundational capabilities for developing medical imaging training workflows in a standardized and reproducible manner. MONAI is important for medical image analysis because it includes specialized tools and functionalities that are tailored to the unique challenges of medical imaging, such as handling 3D data, dealing with various imaging modalities, and integrating with healthcare standards.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip
- virtualenv (optional but recommended)

### Installing TB_ANALYSIS.ipynb

1. Clone the repository:
    ```bash
    git clone https://github.com/JamesMungai254/PowerHacks.git
    cd PowerHacks/TB
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Open the Jupyter Notebook:
    ```bash
    jupyter notebook TB_ANALYSIS.ipynb
    ```

### Installing tbcxrdiagnosis (Django Application)

1. Navigate to the `tbcxrdiagnosis` folder:
    ```bash
    cd tbcxrdiagnosis
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Apply migrations and start the Django server:
    ```bash
    python manage.py migrate
    python manage.py runserver
    ```

5. Open your web browser and navigate to `http://127.0.0.1:8000` to access the application.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.