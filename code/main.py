
#Feburary Week 1 

import pandas as pd
import glob

files = glob.glob('data/Comparison Periods/GTFS-RT data/Feb/Week 1/*.parquet')

feb1_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
feb1_rt_data.head()
feb1_rt_data.columns

#February Week 2

files = glob.glob('data/Comparison Periods/GTFS-RT data/Feb/Week 2/*.parquet')

feb2_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
feb2_rt_data.head()
feb2_rt_data.columns

#April Week 1
files = glob.glob('data/Comparison Periods/GTFS-RT data/April/Week 1/*.parquet')

apr1_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
apr1_rt_data.head()
apr1_rt_data.columns

#April Week 2

files = glob.glob('data/Comparison Periods/GTFS-RT data/April/Week 2/*.parquet')

apr2_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
apr2_rt_data.head()
apr2_rt_data.columns

#July Week 1

files = glob.glob('data/Comparison Periods/GTFS-RT data/July/Week 1/*.parquet')

jul1_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
jul1_rt_data.head()
jul1_rt_data.columns

#July Week 2

files = glob.glob('data/Comparison Periods/GTFS-RT data/July/Week 2/*.parquet')

jul2_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
jul2_rt_data.head()
jul2_rt_data.columns

#October Week 1

files = glob.glob('data/Comparison Periods/GTFS-RT data/October/Week 1/*.parquet')

oct1_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
oct1_rt_data.head()
oct1_rt_data.columns

#October Week 2


files = glob.glob('data/Comparison Periods/GTFS-RT data/October/Week 2/*.parquet')

oct2_rt_data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
oct2_rt_data.head()
oct2_rt_data.columns
