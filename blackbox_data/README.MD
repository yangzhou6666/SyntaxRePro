# Steps to generate pairs on Blackbox server

1. Get an account to access the server.
2. ssh to the server
3. Unzip *mistake-ids.csv.zip*
4. `mkdir java_data`
5. `python combos_to_pair.py` (python2)

Step 5 consumes a large amount of time. The tool we use `/tools/nccb/bin/print-source-state` will output results in the terminal.

The generated result in `java_data` directory will be:

```
java_data
├── fileID_eventID
│   ├── 0
│   ├── 1
|   .....
```


