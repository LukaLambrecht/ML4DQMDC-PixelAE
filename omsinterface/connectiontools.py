#!/usr/bin/env python
# coding: utf-8

# **Tools for checking connectivity to specific URLs and obtaining cookies**  
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
import requests
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree


# local modules
from urls import * # url paths and related settings (e.g. timeout time)
sys.path.append(os.path.abspath('../utils/notebook_utils'))




def check_connectivity(url):
    try:
        requests.get(url, timeout=TIMEOUT_TIME)
        return True
    except (requests.exceptions.ConnectTimeout, requests.exceptions.SSLError) as e:
        return False




def get_cookies(url, authmode, **kwargs):
    if authmode=='certificate':
        if not 'certificate' in kwargs.keys():
            raise Exception('ERROR in connectiontools.py/get_cookies: '
                           +' required argument certificate for mode certificate is missing.')
        certificate = kwargs.pop('certificate')
        return get_cookies_from_certificate(url, certificate, **kwargs)
    elif authmode=='login':
        if not 'login' in kwargs.keys():
            raise Exception('ERROR in connectiontools.py/get_cookies: '
                           +' required argument login for mode login is missing.')
        login = kwargs.pop('login')
        return get_cookies_from_login(url, login, **kwargs)
    else:
        raise Exception('ERROR in connectiontools.py/get_cookies: '
                           +' mode {} not recognized.'.format(mode))




def get_cookies_from_certificate(url, certificate, **kwargs):
    
    print('obtaining cookies for {} from provided certificate {} ...'.format(url,certificates))
    verify = kwargs.pop("verify", None)
    #ca_bundle = certs.where() if verify is None else verify
    ca_bundle = verify
    
    login = kwargs.pop('login', None)

    with requests.Session() as session:
        session.cert = certificate
        #print('certificate:')
        #print(certificate)
        #print('original url:')
        #print(url)

        login_redirect_response = session.get(url, verify=ca_bundle)
        login_redirect_response.raise_for_status()

        redirect_url = login_redirect_response.url
        #print('redirect url:')
        #print(redirect_url)
        authentication_url = _construct_certificate_authentication_url(redirect_url)
        #print('authentication url:')
        #print(authentication_url)

        authentication_response = session.get(authentication_url, verify=ca_bundle)
        authentication_response.raise_for_status()
        #print('response:')
        #print(authentication_response)

        formxml = _modify_xml_content( authentication_response.content )
        action, form_data = _extract_login_form( formxml )
        if login is not None: 
            form_data = {"username":login[0],"password":login[1]}
        print(action)
        print(form_data)
        session.post(url=action, data=form_data, verify=ca_bundle)
        #print('session.cookies:')
        #print(session.cookies)
        
        return session.cookies

def get_cookies_from_login(url, login, **kwargs):
    
    print('obtaining cookies for {} from provided login for username {} ...'.format(url,login[0]))
    verify = kwargs.pop("verify",None)
    
    with requests.Session() as session:

        login_redirect_response = session.get(url, verify=verify)
        login_redirect_response.raise_for_status()
        redirect_url = login_redirect_response.url
        authentication_url = _construct_certificate_authentication_url(redirect_url)

        authentication_response = session.get(authentication_url, verify=verify)
        authentication_response.raise_for_status()

        formxml = _modify_xml_content( authentication_response.content )
        action, form_data = _extract_login_form( formxml )
        form_data = {"username":login[0],"password":login[1]}
        session.post(url=action, data=form_data, verify=verify)

        return session.cookies




def _construct_certificate_authentication_url(login_redirect_url):
    query = urlparse(login_redirect_url).query
    #certificate_authentication_part = "auth/sslclient/" # original
    certificate_authentication_part = "auth" # modification
    base = urljoin(login_redirect_url, certificate_authentication_part)
    return "{}?{}".format(base, query)


def _extract_login_form( xml_response_content ):
    
    tree = ElementTree.fromstring( xml_response_content )
    
    # custom method
    # note: does not seem very generalizable, not sure how to approach...
    form = tree.findall('.//{http://www.w3.org/1999/xhtml}form')
    if len(form)!=1:
        raise Exception('ERROR in connectiontools.py/_extract_login_form: '
                       'login form xml has unexpected format...')
    form = form[0]
    action = form.get("action")
    inputs = tree.findall('.//{http://www.w3.org/1999/xhtml}input')
    form_data = dict(
        (
            (element.get("name"), element.get("value"))
            for element in inputs
        ))
    if len(form_data)<2:
        raise Exception('ERROR in connectiontools.py/_extract_login_form: '
                       'login form xml has unexpected format...')

    # copied code (does not seem to work for current login screen)
    #action = tree.findall("body/form")[0].get("action")
    #form_data = dict(
    #    (
    #        (element.get("name"), element.get("value"))
    #        for element in tree.findall("body/form/input")
    #    )
    #)

    return action, form_data


def _modify_xml_content( xml_response_content ):
    temp = xml_response_content.decode().split('\n')
    for i,line in enumerate(temp):
        if line.strip(' ')[:6] == '<meta ' and line.strip(' ')[-2:] != '/>':
            temp[i] = line.rstrip('>') + '/>'
        if 'autofocus ' in line:
            temp[i] = line.replace('autofocus ','autofocus="off" ')
        if line.strip(' ')=='<hr>':
            temp[i] = ''
        if '<img ' in line:
            temp[i] = line.split('<img ')[0]+'<img '+line.split('<img ')[1].replace('>','/>',1)
    temp = '\n'.join(temp)
    return temp





