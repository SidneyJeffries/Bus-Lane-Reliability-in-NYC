# Bus-Lane-Reliability-in-NYC
Repository for Independent Study Project looking at bus speeds and on time service for select routes within NYC.

**Update September 2nd:**

I was able to find archived GTFS static data through Mobility Database for archived static GTFS data: https://mobilitydatabase.org/feeds/gtfs/mdb-516. I also found GTFS-RT data through for my comparisons period at Bus Observatory API, maintained by Jacobs Urban Tech Club at Cornell: https://api.busobservatory.org/nyct_mta_bus_gtfsrt/schema. This data was saved as separate files by day as parquet files.  I had to change the comparison period from July 2025 to February 2025 due to data availability. The next steps include checking diving deeper into checking the data and double check on how feasible it is to compare this with the rt-data. 


**Update September 10th:**

I read a few scholarly articles regarding on time performance, headway management, and bus lanes within NYC and other major metropolitan cities. The most common articles I found were based on the implementation of bus lanes within cities and the overall benefits of bus service since implementation. Other notable information I found was definitions for on-time performance for NYC busses and other factors that contribute to on-time performance such as route length and use of exclusive bus lanes versus not. My next steps are to load in my GTFS-RT data and see if I can join the trips of the static data. 

**Update September 16th:**

I haven't been able to match the trips just yet, I had some issues working off my mac recently with Python so I'm moving to my PC and migrating the data and code there, as well as updating github with the data and code. I hope to be able to scale up the timeline nedxt week as well as create a list of checks to make sure the data looks good once its joined.
