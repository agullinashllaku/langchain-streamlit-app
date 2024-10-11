import os
import re
import requests
import html2text
import shutil


def delete_folder_if_exists(folder_name):
    # Check if the folder exists in the current directory
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        # Delete the folder and its contents
        shutil.rmtree(folder_name)
        # print(f"Folder '{folder_name}' has been deleted.")
    # else:
    #     print(f"Folder '{folder_name}' does not exist.")


urls = [
    "https://docs.databricks.com/en/machine-learning/index.html",
    "https://docs.databricks.com/en/data-governance/unity-catalog/index.html",
    "https://docs.databricks.com/en/mlflow/tracking.html",
    "https://docs.databricks.com/en/machine-learning/model-serving/index.html",
    "https://docs.databricks.com/en/jobs/index.html",
    "https://docs.databricks.com/en/repos/index.html",
    "https://docs.databricks.com/en/machine-learning/feature-store/index.html",
    "https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html",
    "https://docs.databricks.com/en/pyspark/index.html",
    "https://docs.databricks.com/en/pyspark/basics.html",
    "https://docs.databricks.com/en/pyspark/datasources.html",
    "https://docs.databricks.com/en/pandas/index.html",
    "https://docs.databricks.com/en/pandas/pyspark-pandas-conversion.html",
    "https://docs.databricks.com/en/pandas/pandas-function-apis.html",
    "https://docs.databricks.com/en/udf/pandas.html",
    "https://docs.databricks.com/en/machine-learning/mlops/mlops-workflow.html",
    "https://docs.databricks.com/en/mlflow/tracking.html",
    "https://docs.databricks.com/en/lakehouse-monitoring/custom-metrics.html",
    "https://www.mlflow.org/docs/latest/python_api/mlflow.html",
    "https://docs.databricks.com/clusters/create.html",
    "https://docs.databricks.com/applications/machine-learning/automl.html",
    "https://mlflow.org/docs/latest/tracking.html",
    "https://spark.apache.org/docs/3.5.3/ml-classification-regression.html",
    "https://spark.apache.org/docs/3.5.3/ml-features.html",
    "https://spark.apache.org/docs/3.5.3/ml-tuning.html",
    "https://spark.apache.org/docs/3.5.3/ml-advanced.html",
    "https://www.ibm.com/topics/ensemble-learning",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html",
    "https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/index.html",
    "https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml.html",
    "https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html",
    "https://docs.databricks.com/en/visualizations/index.html",
    "https://docs.databricks.com/en/compute/configure.html",
    "https://www.mlflow.org/docs/latest/model-registry.html",
]


# Function to fetch the HTML content from the URL
def fetch_html_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch {url}: Status code {response.status_code}")


# Function to convert HTML content to markdown
def convert_html_to_markdown(html_content):
    # Initialize html2text converter
    converter = html2text.HTML2Text()
    converter.ignore_links = False  # Keep the links in the markdown
    markdown_content = converter.handle(html_content)
    return markdown_content


def extract_filename_from_url(url):

    match = re.search(r"en/(.*?)(?=\.html)", url)
    if match:
        final_match = match.group(1).replace("/", "-")
        return final_match
    else:

        match = re.search(r"/([^/]+)(?=\.html|$)|/([^/]+)$", url)
        if match:
            if ".html" in match.group(1):
                file_html = match.group(1).replace(".html", "")
                return file_html
            return match.group(1)
        else:
            # Fallback to a generic filename if no pattern is matched
            return "default_filename"


# Function to save markdown content to a file
def save_to_markdown_file(markdown_content, folder_name, filename):
    # Create folder if it doesn't exist
    counter = 1
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if os.path.exists(f"{folder_name}/{filename}.md"):
        full_path = os.path.join(folder_name, f"{filename}_{counter}.md")
        counter += 1
    else:
        full_path = os.path.join(folder_name, filename + ".md")

    with open(full_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_content)


# Define folder where markdown files will be saved
folder_name = "data\markdown"
delete_folder_if_exists(folder_name)

url_dict = {}

for i, url in enumerate(urls):
    try:
        # Fetch HTML content from the URL
        html_content = fetch_html_content(url)

        # Convert HTML to markdown format
        markdown_content = convert_html_to_markdown(html_content)

        # Extract a clean filename from the URL
        filename = extract_filename_from_url(url)

        # Save the markdown content to the file
        save_to_markdown_file(markdown_content, folder_name, filename)
        saved_path = os.path.join(folder_name, filename + ".md")
        url_dict[saved_path] = url

    except Exception as e:
        print(f"Failed to process {url}: {e}")
