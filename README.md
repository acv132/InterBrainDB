
# InterBrainDB - Living Literature Review Database on Hyperscanning

## Purpose
This repository is dedicated to hosting a living literature review tracking emerging research on multimodal 
hyperscanning in contexts with a digital component. The platform includes various categories, such as interaction scenario, type of tasks, measurement 
modalities, and analysis approaches, serving as a dynamic open-access resource. 
The aim is to provide a comprehensive overview of the current state of hyperscanning research, 
enabling researchers to stay informed about the latest developments in the field.

The code is based on the [streamlit](https://streamlit.io/) library.

## Living Literature Review Database on Hyperscanning
Find the server-hosted version of the living review here:
[Link]()

## Original Paper
Read the original literature review here:
[Link Button]()
```
[citation with doi]
```
If you use this resource, please consider citing our paper. 

## Hosting the App Locally
1. Clone the repository
```bash
git clone https://github.com/acv132/InterBrainDB
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
4. Use the `Download CSV` button in the Database tab to download the latest version of the database as a CSV file. 
   Hint: select the radio button "All Columns" to get the full database and avoid errors.
5. Copy the `database.csv` file into the `data` folder.
6. Start the Streamlit app
```bash
streamlit run .\app.py --server.port 8501
```
7. You can now view the streamlit app in your browser via a local URL: 
http://localhost:8501
