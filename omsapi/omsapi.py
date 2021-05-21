# pylint: disable=W0702,R0902
""" CMS OMS Aggregation API python client
"""

from __future__ import print_function
import os
import requests
import subprocess
import json
import time

# Suppress InsecureRequestWarning
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests.exceptions import ConnectionError
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


OMS_FILTER_OPERATORS = ["EQ", "NEQ", "LT", "GT", "LE", "GE", "LIKE"]
OMS_INCLUDES = ["meta", "presentation_timestamp", "data_only"]

#OpenID parameters
cern_auth_token_url='https://auth.cern.ch/auth/realms/cern/protocol/openid-connect/token'
grant_type='client_credentials'
exc_token_type='access_token'

class OMSApiException(Exception):
    """ OMS API Client Exception """
    pass


class OMSQuery(object):
    """ OMS Query object """

    def __init__(self, base_url, resource, verbose, cookies, oms_auth, cert_verify, retry_on_err_sec, proxies):
        self.attribute_validation = True
        self.base_url = base_url
        self.resource = resource
        self.verbose = verbose
        self.cookies = cookies
        self.oms_auth = oms_auth
        self.cert_verify = cert_verify
        self.err_sec = retry_on_err_sec
        self.proxies = proxies

        self._attrs = None  # Projection
        self._filter = []  # Filtering
        self._sort = []  # Sorting
        self._include = []  # Include
        self._custom = []  # Custom parameters: array of key:value

        # Pagination
        self.page = 1
        self.per_page = 10

        # Metadata
        self.metadata = None

        self._load_meta()

    def _attr_exists(self, attr):
        """ Check if attribute exists

            Returns:
                bool: False if attribute does not exist
                      True if exists or it is not available to check
        """

        if self.metadata and attr not in self.metadata:
            self._warn("Attribute [{attr}] does not exist. " +
                       "Check for a typo or disable validation " +
                       "by .set_validation(False) ".format(attr=attr))

            # Return True if attribute validation is disabled
            return False == self.attribute_validation

        return True

    def _load_meta(self):
        """ Load meta information about resource without fetching data"""

        resourceBase = self.resource.split("/")[0]
        url = "{base_url}/{resource}/meta".format(base_url=self.base_url,
                                                  resource=resourceBase)

        response = self.get_request(url, verify=self.cert_verify)

        if response.status_code != 200:
            self._warn("Failed to fetch meta information")
        else:
            try:
                self.metadata = response.json()["meta"]["fields"]
            except (ValueError, KeyError, TypeError):
                self._warn("Meta information is incorrect")

    def _warn(self, message, raise_exc=False):
        """ Print Warning message or raise Exception

            Args:
                message (str): warning message to be printed or raised as exception
                raise_exc (bool):  raise Exception or just print warning
        """

        if raise_exc:
            raise OMSApiException(message)

        if self.verbose:
            print("Warning: {message}".format(message=message))

    def set_verbose(self, verbose):
        """ Set verbose flag

            Args:
                verbose (bool): True/False

            Examples:
                .set_verbose(True)
        """
        self.verbose = verbose

    def set_validation(self, attribute_validation):
        """ Enable or disable attribute validation for
            filtering/sorting/projection

            Args:

                attribute_validation (bool): True/False

            Examples:
                .set_validation(False)
        """

        self.attribute_validation = attribute_validation

    def attrs(self, attributes=None):
        """ Projection. Query only those attributes which you need.

            Args:
                attributes (list): list of attribute names.

            Examples:
                .attrs(["fill_number", "run_number"])
        """

        if not isinstance(attributes, list):
            self._warn("attrs() - attributes must be a list", raise_exc=True)

        # Find only existing attributes, remove duplicates
        self._attrs = [attr for attr in set(
            attributes) if self._attr_exists(attr)]

        return self

    def filters(self, filters):
        """ Filtering of a result set. Apply list of filters.
            Same as calling .filter(attribute_name, value, operator) multiple times

            Args:
                filters (list): list of filters (dicts) [{"attribute_name": X, "value": Y, "operator": Z}, ...]
        """

        for f in filters:
            self.filter(f["attribute_name"], f["value"], f["operator"])

    def filter(self, attribute, value, operator="EQ"):
        """ Filtering of a result set. Apply single filter

            Args:
                attribute (str): name of a attribute (field)
                value (str/int/bool): filtering against value
                operator (str): one of supported operators (OMS_FILTER_OPERATORS)

            Examples:
                .filter("fill_number", 5000, "GT")
        """

        if not isinstance(operator, str):
            self._warn("filter() - operator name must be a string",
                       raise_exc=True)

        operator = operator.upper()

        if operator not in OMS_FILTER_OPERATORS:
            self._warn("filter() - [{op}] is not supported operator".format(op=operator),
                       raise_exc=True)

        if self._attr_exists(attribute):
            # Check metadata if attribute is searchable
            searchable = True
            try:
                searchable = self.metadata["searchable"]
            except (KeyError, TypeError):
                # Metadata is not available or not complete
                pass

            if searchable:
                self._filter.append("filter[{k}][{op}]={v}".format(k=attribute,
                                                                   op=operator,
                                                                   v=value))
        return self

    def clear_filter(self):
        """ Remove all filters
        """

        self._filter = []

        return self

    def sort(self, attribute, asc=True):
        """ Sort result set

            Args:
                attribute (str): name of attribute (field)
                asc (bool): ascending direction or not

            Examples:
                .sort("fill_number", asc=False)
        """

        if not isinstance(attribute, str):
            self._warn("sort() - attribute name must be a string",
                       raise_exc=True)

        if self._attr_exists(attribute):
            # Check metadata if attribute is sortable
            sortable = True
            try:
                sortable = self.metadata["fields"]["sortable"]
            except (KeyError, TypeError):
                # Metadata is not available or not complete
                pass

            if sortable:
                if not asc:
                    attribute = "-" + attribute

                self._sort.append(attribute)

        return self

    def paginate(self, page=1, per_page=10):
        """ Paginate result set

            Args:
                page (int): page number
                per_page (int): page size(limit)

            Examples:
                .paginate(2, 25)
        """

        self.page = page
        self.per_page = per_page

        return self

    def include(self, key):
        """ Include special flags to a query

            Args:
                key (str): one of supported keys

            Examples:
                .include("meta")
        """

        if key not in OMS_INCLUDES:
            self._warn("{key} is not supported include".format(key=key),
                       raise_exc=True)

        self._include.append(key)

        return self

    def custom(self, key, value=None):
        """ Add custom parameter (key-value pair) in a query

            Args:
                key(str): custom parameter name
                value(str/int/bool): custom parameter value

            Examples:
                .custom("group[size]", 100)
        """

        self._custom.append("{k}={v}".format(k=key, v=value))

        return self

    def data_query(self):
        """ Contruct URL to be used to query data from API

            Returns:
                str
        """

        url = "{base_url}/{resource}/".format(base_url=self.base_url,
                                              resource=self.resource)

        url_params = []

        # Project
        if self._attrs:
            url_params.append("fields=" + ",".join(set(self._attrs)))

        # Filter
        url_params.extend(self._filter)

        # Sort
        if self._sort:
            url_params.append("sort=" + ",".join(set(self._sort)))

        # Include
        if self._include:
            url_params.append("include=" + ",".join(set(self._include)))

        # Paginate
        page_offset = self.per_page * (self.page - 1)
        url_params.append("page[offset]={offset}".format(offset=page_offset))
        url_params.append(
            "page[limit]={per_page}".format(per_page=self.per_page))

        # Custom parameters
        url_params.extend(self._custom)

        if url_params:
            url = url + "?" + "&".join(url_params)

        return url


    def data(self):
        """ Execute query and retrieve data

            Returns:
                requests.Response object
        """

        url = self.data_query()

        if self.verbose:
            print(url)
        if self.err_sec > 0:
            while True:
                try:
                    ret = self.get_request(url, verify=self.cert_verify)
                    break
                except ConnectionError as ex:
                    print("Warning: will retry in " + str(self.err_sec) + "seconds after connection error: " + str(ex))
                    time.sleep(self.err_sec)
        else:
            ret = self.get_request(url, verify=self.cert_verify)

        return ret

    def meta(self):
        """ Returns metadata of a resource.

            Returns:
                str: if metadata is available
                None: if metadata is unavailable
        """
        return self.metadata

    def get_request(self, url, verify=False):
        if self.oms_auth:
            response = requests.get(url, verify=verify, headers=self.oms_auth.token_headers, proxies=self.proxies)
            #check if token has expired (Unauthorized)
            if response.status_code == 401:
                print("Unauthorized. Will try to obtain a new token")
                self.oms_auth.auth_oidc()
                return requests.get(url, verify=verify, headers=self.oms_auth.token_headers, proxies=self.proxies)
            return response
        else:
            return requests.get(url, verify=verify, cookies=self.cookies, proxies=self.proxies)
 
class OMSAPIOAuth(object):
    """ OMS API token store and manager """

    def __init__(self, client_id, client_secret, audience="cmsoms-prod", cert_verify=True, proxies={}, retry_on_err_sec=0):
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience
        self.cert_verify = cert_verify
        self.proxies = proxies
        self.token_json = None
        self.token_time = None
        self.err_sec = retry_on_err_sec
 
    def auth_oidc(self):
        """ Authorisation Using CERN Open ID authentication wrappeer"""
        if self.err_sec > 0:
            while True:
                try:
                    ret = self.auth_oidc_req()
                    return ret
                except ConnectionError as ex:
                    print("Warning: will retry auth_oidc in " + str(self.err_sec) + "seconds after connection error: " + str(ex))
                    time.sleep(self.err_sec)
        else:
            return self.auth_oidc_req()

    def auth_oidc_req(self):
        """ Authorisation Using CERN Open ID authentication """

        current_time = time.time()
        if self.token_json and self.token_time:
            if current_time - self.token_time < 30:
                print("Warning: token was requested less than 30 seconds ago. Will not renew this time.")
                return
                
        self.token_time = current_time
        token_req_data = {
            'grant_type': grant_type,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        ret = requests.post(cern_auth_token_url, data=token_req_data, verify=self.cert_verify, proxies=self.proxies)
        if ret.status_code!=200:
            raise Exception("Unable to acquire OAuth token: " + ret.content.decode())

        res = json.loads(ret.content)

        exchange_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'subject_token': res['access_token'],
            'audience': self.audience,
            'grant_type':'urn:ietf:params:oauth:grant-type:token-exchange',
            'requested_token_type':'urn:ietf:params:oauth:token-type:'+ exc_token_type
        }

        #cert verification disabled
        ret = requests.post(cern_auth_token_url, data=exchange_data, verify=self.cert_verify, proxies=self.proxies)
        if ret.status_code!=200:
            raise Exception("Unable to exchange OAuth token: " + ret.content.decode())

        self.token_json = json.loads(ret.content)
        self.token_headers = {'Authorization':'Bearer ' + self.token_json["access_token"]}

 
class OMSAPI(object):
    """ Base OMS API client """

    def __init__(self, api_url="https://cmsoms.cern.ch/agg/api", api_version="v1", verbose=True, cert_verify=True, retry_on_err_sec=0, proxies={}):
        self.api_url = api_url
        self.api_version = api_version
        self.verbose = verbose
        self.cert_verify = cert_verify 
        self.err_sec = retry_on_err_sec
        self.proxies = proxies

        self.base_url = "{api_url}/{api_version}".format(api_url=api_url,
                                                         api_version=api_version)

        index =  self.api_url.find('://')
        if index >= 0:
            tmp = self.api_url[index + 3 : ]
            self.api_url_host = self.api_url[: index + 3] + tmp[: tmp.find('/')]
        else:
            self.api_url_host = self.api_url_host[: self.api_url_host.find('/')]
        self.oms_auth = None
        self.cookies = {}

    def query(self, resource, query_validation=True):
        """ Create query object """

        q = OMSQuery(self.base_url, resource=resource, verbose=self.verbose,
                     cookies=self.cookies, oms_auth=self.oms_auth, cert_verify=self.cert_verify, retry_on_err_sec=self.err_sec, proxies=self.proxies)

        return q

    def auth_oidc(self, client_id, client_secret, audience="cmsoms-prod", proxies={}):
        """ Authorisation Using CERN Open ID authentication """

        if not self.oms_auth:
            self.oms_auth = OMSAPIOAuth(client_id, client_secret, audience, self.cert_verify, proxies=proxies, retry_on_err_sec=self.err_sec)
        self.oms_auth.auth_oidc()

    def auth_krb(self, cookie_path="ssocookies.txt"):
        """ Authorisation for https using kerberos"""

        def rm_file(filename):
            if os.path.isfile(filename) and os.access(filename, os.R_OK):
                os.remove(filename)

        rm_file(cookie_path)
        args = ["auth-get-sso-cookie", "-u", self.api_url_host, "-o", cookie_path]
        if not self.cert_verify:
            args.append("--nocertverify")
        try:
            subprocess.call(args)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                #this package is available from CERN repos:
                #http://linuxsoft.cern.ch/internal/repos/authz7-stable/x86_64/os
                #http://linuxsoft.cern.ch/internal/repos/authz8-stable/x86_64/os
                raise OMSApiException(
                    "Required package is not available. yum install auth-get-sso-cookie")
            else:
                raise OMSApiException("Failed to authenticate with kerberos")

        self.cookies = {}

        with open(cookie_path, "r") as f:
            cookies_raw = f.read()

            for line in cookies_raw.split("\n"):
                fields = line.split()
                if len(fields) == 7:
                    # fields = domain tailmatch path secure expires name value
                    key = fields[5]

                    if any(p in key for p in ["mod_auth_openidc_session"]):
                        self.cookies[key] = fields[6]

        if not self.cookies.keys():
            raise OMSApiException("Unkown cookies")

        rm_file(cookie_path)
