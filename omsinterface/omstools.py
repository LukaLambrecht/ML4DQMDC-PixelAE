#!/usr/bin/env python
# coding: utf-8

# **Tools for accessing the OMS database**  
# 
# The functions in this script are not my own, but largely based on the wbmcrawler and cernrequests packages.  
# See the readme file in this directory for more information.  
# 
# For normal users these functions should not be called directly, everything is handled by a single call to get_oms_data.py / get_oms_data.  
# See get_oms_data.py in this directory for more information.



### imports

# external modules
import sys
import os
import math
import requests
import importlib
from urllib.parse import urlencode


# local modules
from urls import * # url paths and related settings (e.g. timeout time)
import connectiontools
importlib.reload(connectiontools)
from connectiontools import check_connectivity, get_cookies
sys.path.append(os.path.abspath('../utils/notebook_utils'))




def check_oms_connectivity():
    return check_connectivity(OMS_API_URL)




def get_oms_cookies( authmode, **kwargs ):
    url = OMS_ALTERNATIVE_API_URL
    return get_cookies( url, authmode, **kwargs )




PAGE_SIZE = 1000

def _get_oms_resource_within_cern_gpn(relative_url):
    url = "{}{}".format(OMS_API_URL, relative_url)
    return requests.get(url)


def _get_oms_resource_authenticated(relative_url, cookies):
    url = "{}{}".format(OMS_ALTERNATIVE_API_URL, relative_url)
    return requests.get(url, cookies=cookies, verify=False)


def get_oms_resource(table, parameters, **kwargs):
    parameters = urlencode(parameters)
    relative_url = "{table}?{parameters}".format(table=table, parameters=parameters)
    print('relative url: '+str(relative_url))
    
    inside_cern_gpn = check_oms_connectivity()
    if inside_cern_gpn:  # Within CERN GPN
        response = _get_oms_resource_within_cern_gpn(relative_url)
    else:  # Outside CERN GPN, requires authentication
        if not 'authmode' in kwargs.keys():
            raise Exception('ERROR in omstools.py/get_oms_resource: '
                           +'need authentication parameters')
        authmode = kwargs.pop('authmode')
        cookies = get_oms_cookies( authmode, verify=False, **kwargs )
        response = _get_oms_resource_authenticated(relative_url, cookies)
        
    return response.json()


def _get_single_resource(table, parameters, **kwargs):
    if "inside_cern_gpn" not in kwargs:
        kwargs['inside_cern_gpn'] = check_oms_connectivity()
    data = get_oms_resource(table, parameters, **kwargs)["data"]
    assert len(data) == 1, "More than 1 {} were returned".format(table)
    return data[0]


def get_run(run_number, **kwargs):
    parameters = {"filter[run_number][EQ]": run_number, "sort": "-run_number"}
    return _get_single_resource("runs", parameters, **kwargs)


def get_fill(fill_number, **kwargs):
    parameters = {"filter[fill_number][EQ]": fill_number, "sort": "-fill_number"}
    return _get_single_resource("fills", parameters, **kwargs)


def _get_resources_page(table, parameters, page, page_size, **kwargs):
    assert page >= 1, "Page number cant be lower than 1"
    params = {"page[offset]": (page - 1) * page_size, "page[limit]": page_size}
    params.update(parameters)
    
    oms_resource = get_oms_resource(table, params, **kwargs)

    return get_oms_resource(table, params, **kwargs)


def get_resources(table, parameters, page_size=PAGE_SIZE, silent=False, **kwargs):
    if "inside_cern_gpn" not in kwargs:
        kwargs['inside_cern_gpn'] = check_oms_connectivity()

    if not silent:
        print("Getting initial response...", end="\r")

    response = _get_resources_page(
        table, parameters, page=1, page_size=page_size, **kwargs
    )
    
    resource_count = response["meta"]["totalResourceCount"]
    page_count = calc_page_count(resource_count, page_size)

    if not silent:
        print(" " * 100, end="\r")
        print("Total number of {}: {}".format(table, resource_count))
        print()

    resources = [flatten_resource(resource) for resource in response["data"]]

    for page in range(2, page_count + 1):
        if not silent:
            print_progress(page, page_count, text="Page {}/{}".format(page, page_count))
        response = _get_resources_page(table, parameters, page, page_size, **kwargs)
        resources.extend([flatten_resource(resource) for resource in response["data"]])

    if not silent:
        print()
        print()

    assert len(resources) == resource_count, "Oops, not enough resources were returned"
    return resources


def get_runs(begin, end, **kwargs):
    print("Getting runs {} - {} from CMS OMS".format(begin, end))
    parameters = {
        "filter[run_number][GE]": begin,
        "filter[run_number][LE]": end,
        "filter[sequence][EQ]": "GLOBAL-RUN",
        "sort": "run_number",
    }

    return get_resources("runs", parameters, page_size=100, **kwargs)


def get_fills(begin, end, **kwargs):
    print("Getting fills {} - {} from CMS OMS".format(begin, end))
    parameters = {
        "filter[fill_number][GE]": begin,
        "filter[fill_number][LE]": end,
        "sort": "fill_number",
    }

    split_scheme = kwargs.pop("split_filling_scheme", False)

    if not split_scheme:
        return get_resources("fills", parameters, page_size=100, **kwargs)
    else:
        fills = get_resources("fills", parameters, page_size=100, **kwargs)
        for fill in fills:
            split_filling_scheme(fill)
        return fills


def get_lumisection_count(run_number, **kwargs):
    """
    :return: Number of lumisections where CMS was active
    """
    parameters = {
        "filter[run_number][EQ]": run_number,
        #"filter[cms_active][EQ]": 'true', # removed due to changes in oms api not supporting this anymore
        "sort": "lumisection_number"
    }

    if "inside_cern_gpn" not in kwargs:
        kwargs['inside_cern_gpn'] = check_oms_connectivity()

    response = _get_resources_page(
        "lumisections", parameters, page_size=1, page=1, **kwargs
    )
    resource_count = response["meta"]["totalResourceCount"]
    return resource_count


def get_lumisections( run_number=None, fill_number=None, start_time=None, end_time=None, **kwargs):
    assert (
        bool(run_number) ^ bool(fill_number) ^ bool(start_time and end_time)
    ), "Specify either run number or fill number or time range"

    parameters = {}
    if run_number:
        parameters["filter[run_number][EQ]"] = run_number
    elif fill_number:
        parameters["filter[fill_number][EQ]"] = fill_number
    elif start_time and end_time:
        parameters["filter[start_time][GE]"] = start_time
        parameters["filter[end_time][LE]"] = end_time
    parameters["sort"] = "lumisection_number"

    return get_resources("lumisections", parameters, page_size=5000, **kwargs)


def get_hltpathinfos(run_number, **kwargs):
    parameters = {"filter[run_number][EQ]": run_number}

    return get_resources("hltpathinfo", parameters, page_size=1000, **kwargs)


def get_hltpathrates(run_number, path_name, **kwargs):
    parameters = {
        "filter[last_lumisection_number][GT]": 0,
        "filter[path_name][EQ]": path_name,
        "filter[run_number][EQ]": run_number,
        "sort": "last_lumisection_number",
        "group[granularity]": "lumisection",
    }
    return get_resources("hltpathrates", parameters, page_size=10000, **kwargs)


def get_all_hltpathrates(run_number, silent=False, **kwargs):
    if not silent:
        print("Retrieving all hltpathrates for run number {}".format(run_number))
        print("Getting list of available hltpathinfos...")

    hltpathinfos = get_hltpathinfos(run_number, silent=silent, **kwargs)

    path_info_count = len(hltpathinfos)

    path_names = [pathinfo["path_name"] for pathinfo in hltpathinfos]

    hltpathrates = []

    for i, path_name in enumerate(path_names, 1):

        print_progress(
            i,
            path_info_count,
            text="Path {}/{}: {:80s}".format(i, path_info_count, path_name),
        )
        hltpathrates.extend(
            get_hltpathrates(run_number, path_name, silent=silent, **kwargs)
        )

    return hltpathrates




def calc_page_count(resource_count, page_size):
    return math.ceil(resource_count / page_size)


def flatten_resource(response):
    response_flat = response["attributes"]
    if response["type"] in ["runs", "fills"] and "meta" in response:
        for key, value in response["meta"]["row"].items():
            new_field_name = "{}_unit".format(key)
            response_flat.update({new_field_name: value["units"]})
    return response_flat


def progress_bar(current, total, text="", filler="#"):
    bar_length = 50
    processed = current / total
    processed_length = math.ceil(processed * bar_length)

    bar = filler * processed_length + "-" * (bar_length - processed_length)

    return "[{}] {:.2%} {}".format(bar, processed, text)


def print_progress(current, total, text="", *args, **kwargs):
    print(progress_bar(current, total, text, *args, **kwargs), end="\r")
    sys.stdout.flush()










