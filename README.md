# Bus-Lane-Reliability-in-NYC
Repository for Independent Study Project looking at bus speeds and on time service for select routes within NYC.

[Add more about how the code generally works, workflows, methodology, and analyis]
- Add the specifics about what the on-time thresholds are,the comparion periods, the data sources, and high level results
- Mention more about the specifics of the prediction model once its flushed out

#Files in the Main Branch   
gtfs.py: Builds the historic static schedule by loading in the static gtfs data parses it into the two week comparison periods for February, July, April, and October.

gtfsrt.py: Loads the GTFS-RT data into parses them by comparison period

join_static.py: Joins MTA real-time bus GPS data to the static schedule, then estimates stop-by-stop arrivals for Brooklyn routes. It uses those arrivals to calculate headways, bunching, and travel times for each comparison period.

flatbush_vs_fulton.py: This file does the actual comparison between the Fulton bus corridor and Flatbush bus corridor. The on-time metric, charts, and maps comparing the two live in here. 

rtintervals.py: Measures how often each bus reports its location by calculating the time between consecutive Automatic Vehicle Location (AVL) updates.

schedule_status.py: Classifies on-time performance as early, on-time, and late by using thresholds of 1 minute or earlier and 5 minutes or more for late. 

#Files in the Main Branch   



