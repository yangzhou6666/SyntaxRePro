#! /usr/bin/env python
import os, subprocess, sys, tempfile
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
        tff, tfp = tempfile.mkstemp()
        os.close(tff)
        while True:
            l = q.get()
            i = done.add(1)
            if i % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            file_id = l[0]
            fail_event_id = l[1]
            success_event_id = l[2]
            rtn = os.system("/tools/nccb/bin/print-source-state %s %s > %s" %(file_id, fail_event_id, tfp))
            if rtn != 0:
                q.task_done()
                continue
            rtn = os.system("../error_recovery_experiment/blackbox/grmtools/target/release/lrlex ../error_recovery_experiment/blackbox/grammars/java7/java.l %s > /dev/null 2> /dev/null" % tfp)
            if rtn != 0:
                q.task_done()
                continue

            out = subprocess.check_output(["../error_recovery_experiment/runner/java_parser_none", tfp])
            if "Parsed successfully" in out:
                with open(tfp, mode='r') as code:
                    content = code.read()
                    errors = java.check_syntax(content)
                    if len(errors) > 0:
                        with open('a.txt', mode='a') as f:
                            f.write(file_id)
                            f.write('\n')

                            if errors[0][2] == "= expected":
                                f.write(errors[0][3])
                                f.write(content)
                            f.write(errors[0][2])
                            f.write('\n')
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
