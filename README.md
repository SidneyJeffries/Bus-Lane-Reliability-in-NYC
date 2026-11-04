# Bus-Lane-Reliability-in-NYC
Repository for Independent Study Project looking at bus speeds and on time service for select routes within NYC.
  
gtfs.py: Builds the historic static schedule by loading in the static gtfs data parses it into the two week comparison periods for February, July, April, and October.
gtfsrt.py:
join_static.py:
flatbush_vs_fulton.py: This file does the actual comparison between the Fulton bus corridor and Flatbush bus corridor. The on-time metric, charts, and maps comparing the two live in here. 
rtintervals.py: Checks the intervals of 
schedule_status.py: Classifies on-time performance as early, on-time, and late by using thresholds of 1 minute or earlier and 5 minutes or more for late. 






