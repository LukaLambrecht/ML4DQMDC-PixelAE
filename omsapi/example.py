####################################################
# Similar to example.ipynb but in plain .py format #
####################################################

### imports

# external modules
import matplotlib.pyplot as plt

# local modules
import get_oms_data
from get_oms_data import get_oms_api, get_oms_data, get_oms_response_attribute

# get the omsapi instance
omsapi = get_oms_api()

# example: get run information for a single run
runnb = 297050
run_info = get_oms_data( omsapi, 'runs', runnb=runnb )
print(run_info)
