#####################
# submit a test job #
#####################

import os
import sys
import condortools as ct


def dosomething():
    ### do something in a job
    print('it worked!')

if __name__=='__main__':
    # command line arguments: 
    # run without any command line arguments to submit a job;
    # add "local" on the command line to run locally

    runmode = 'condor'

    if len(sys.argv)>1:
        if len(sys.argv)==2 and sys.argv[1]=='local':
            runmode = 'local'
        else:
            raise Exception('ERROR: unrecognized command line args: {}'.format(sys.argv))

    if runmode=='condor':
        cmd = 'python submit_test_job.py local'
        ct.submitCommandAsCondorJob('cjob_submit_test_job', cmd)

    elif runmode=='local':
        dosomething()
