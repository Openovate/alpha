# -->

import os
import sys
import urllib
import urlparse
import hashlib
import shutil
import json
import simplejson as sjson

from itertools import product

# get root path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# get data path
def get_data_path(site):
    # raw extracted data path
    # page screenshot path (debugging)
    # clustered data path
    # trained data path
    paths = {
        'root'          : os.path.join(root, 'data', site),
        'raw'           : os.path.join(root, 'data', site, 'raw'),
        'screenshot'    : os.path.join(root, 'data', site, 'screenshot'),
        'cluster'       : os.path.join(root, 'data', site, 'cluster'),
        'trained'       : os.path.join(root, 'data', site, 'trained')
    }

    return paths

# load raw data
def load_raw_data(site, id):
    # get the raw path
    path = get_data_path(site)['raw']

    # append site id
    path = os.path.join(path, id + '.json')

    data = {}

    # open file
    with open(path, 'r') as f:
        data = json.load(f)

    return data

# load clustered data
def load_clustered_data(site):
    return ''

# load trained data
def load_trained_data(site):
    return ''

# load urls
def load_urls(site):
    # get root path
    path = get_data_path(site)['root']
    # get urls path
    urls = os.path.join(path, 'urls.json')

    # path exists?
    if not os.path.exists(path):
        # create directory
        os.makedirs(path)

    # file eixsts?
    if not os.path.exists(urls):
        # create file
        f = file(urls, 'w+')
        # close handler
        f.close()

    # open up urls file
    with open(urls, 'r+') as f:
        try:
            return json.load(f)
        except:
            return {}

# save url
def save_url(site, url, id):
    # get root path
    path = get_data_path(site)['root']
    # get urls path
    urls = os.path.join(path, 'urls.json')

    # json data
    data = []

    # write?
    write = True

    # load the urls file
    with open(urls, 'r+') as f:
        try:
            # load data
            data = json.load(f)

            # data exists?
            if id in data:
                write = False
                return

            # append data
            data[id] = { 'url' : url, 'id' : id }
        except:
            # create data
            data = { id : { 'url' : url, 'id' : id } }

    # open file for writing
    with open(urls, 'w+') as f:
        # do we need to write?
        if not write:
            return

        f.write(json.dumps(data, indent=2, ensure_ascii=False).encode('utf8'))

# clean site data
def clean_data(site):
    # get root site data
    path = get_data_path(site)['root']

    # path exists?
    if os.path.exists(path):
        # remove entire path
        shutil.rmtree(path)

# url parser
def parse_url(site):
    return urlparse.urlparse(site)

# generate id
def generate_id(id):
    return hashlib.md5(id).hexdigest()

# pretty print object
def pretty(object):
    print json.dumps(object, indent=2).encode('utf8')

# consolidate / normalize selectors
def consolidate_selectors(selectors):
    for selector1, selector2 in product(selectors, repeat=2):
        if selector1 is selector2:
            continue

        # element tag name needs to match
        names1 = ' > '.join([s['name'] for s in selector1])
        names2 = ' > '.join([s['name'] for s in selector2])

        if names1 != names2:
            continue

        for part1, part2 in zip(selector1, selector2):
            if part1['id'] != part2['id']:
                part1['id'] = ''
                part2['id'] = ''

            classes = list(set(part1['classes']) & set(part2['classes']))

            part1['classes'] = classes
            part2['classes'] = classes

    consolidated = dict()

    for selector in selectors:
        paths = []

        for part in selector:
            path = part['name']

            if part['id']:
                path += '#' + part['id']
            if part['classes']:
                path += '.' + '.'.join(part['classes'])
            
            paths.append(path)
        
        consolidated[' > '.join(paths)] = selector

    return consolidated

