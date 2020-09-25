#! /usr/bin/env python

import os, sys
from Queue import Queue
from threading import Thread
import numpy as np
import javac_parser
from atomicint import AtomicInt


q = Queue(512)
done = AtomicInt(0)
class Worker(Thread):
    def run(self):
        java = javac_parser.Java()
        i = 0
        while True:
            l = q.get()
            i = done.add(1)
            if i % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            file_id = l[0]
            fail_event_id = l[1]
            success_event_id = l[2]
            os.system("mkdir ./java_data/%s_%s" % (fail_event_id, success_event_id))
            os.system("/tools/nccb/bin/print-source-state %s %s > ./java_data/%s_%s/0" % (file_id, fail_event_id, fail_event_id, success_event_id))
            os.system("/tools/nccb/bin/print-source-state %s %s > ./java_data/%s_%s/1" % (file_id, success_event_id, fail_event_id, success_event_id))

            q.task_done()

for _ in range(12):
    w = Worker()
    w.daemon = True
    w.start()

with open('mistake-ids.csv', 'r') as f:
    for pair in f:
        q.put(pair.strip().split(','))
    q.join()
    print
