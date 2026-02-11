
# InterBrainDB - Living Literature Review Database on Hyperscanning

## Purpose
This repository is dedicated to hosting a living literature review tracking emerging research on multimodal 
hyperscanning in contexts with a digital component. The platform includes various categories, such as interaction scenario, type of tasks, measurement 
modalities, and analysis approaches, serving as a dynamic open-access resource. 
The aim is to provide a comprehensive overview of the current state of hyperscanning research, 
enabling researchers to stay informed about the latest developments in the field.

The code is based on the [streamlit](https://streamlit.io/) library.

## Living Literature Review Database on Hyperscanning
Find the server-hosted version of the living review here: [InterBrainDB](https://websites.fraunhofer.de/interbraindb/)

## Original Paper
Read the original literature review here:
[Frontiers in Neuroergonomics](https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2026.1756956/full)
```bibtex
@article{Vorreuther2026Feb,
  author  = {Vorreuther, Anna and Brouwer, Anne-Marie and Vukelić, Mathias},
  title   = {Reviewing digital collaborative interactions with multimodal hyperscanning through an ever-growing database},
  journal = {Front. Neuroergonomics},
  volume  = {7},
  pages   = {1756956},
  year    = {2026},
  month   = {feb},
  issn    = {2673-6195},
  publisher = {Frontiers},
  doi     = {10.3389/fnrgo.2026.1756956}
}
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
4. Use the `Download CSV` button in the Database tab of the server-hosted version to download the latest version of the database as a CSV file. 
   Hint: select the radio button "All Columns" to get the full database and avoid errors.
5. Copy the `database.csv` file into the `data` folder.
6. Start the Streamlit app
```bash
streamlit run .\app.py --server.port 8501
```
7. You can now view the streamlit app in your browser via a local URL: 
http://localhost:8501
