import sys
import os
import shutil
from utils.args import Parser3rdParty

ARGS = {
    "filename": str,
    "taskname": str,
    
}

parser = Parser3rdParty(ARGS)

BASE_TASK = 'task/base.yaml'

shutil.copyfile(BASE_TASK, "new_task.yaml")