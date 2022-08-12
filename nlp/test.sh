#!/bin/bash

set -ex
python baselineflow.py run
python branchflow.py run
python nlpflow.py run
python predflow.py run
