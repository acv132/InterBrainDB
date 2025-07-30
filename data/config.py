import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
data_dir = Path(os.getenv('data_dir'))
file = os.getenv('file')

# plots and visuals
# ColorMap = ['#00928f', '#00567c', '#6d90a0', '#38b6c0', '#728128', '#bad23c', '#81c5cb', '#19bdff', '#bcbec0', ]
ColorMap = ["#ADE1DC", "#C5C2E4", "#F28C8C", "#9FCBE1", "#F8BC63", "#CBEA7B", "#F5C6DA", "#BFD8B8", "#F9D9B6"]
