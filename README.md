
# Living Literature Review on Hyperscanning with Digital Components

## Purpose
This repository is dedicated to hosting a living literature review tracks emerging research on multimodal hyperscanning in contexts with a digital 
component. The platform includes various categories, such as interaction scenario, type of tasks, measurement 
modalities, and analysis approaches, serving as a dynamic open-access resource. 
The aim is to provide a comprehensive overview of the current state of digital hyperscanning research, 
enabling researchers to stay informed about the latest developments in the field.

The code is based on the [streamlit](https://streamlit.io/) library.

## Living Literature Review on Hyperscanning with Digital Components
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
git clone gitlink_to_your_repo.git
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Start the Streamlit app
```bash
streamlit run .\src\Welcome.py --server.port 8501
```
4. You can now view the streamlit app in your browser via a local URL: 
http://localhost:8501


### TODO
- add respiration as own category measurement modality
- add fMRI as own category measurement modality
- add more details to "included" studies through other_labels 
  - double-check number of fNIRS channels and number of EEG-channels info exists for all of them