#!/bin/env python
"""
Purpose:    parse configuration file for datacube workflow to kickoff.

Input:      User input path2aoi.yml
Output:     
Assumption: 
ProcessLogics :

Usage:      python loadconf.py path2aoi.yml 
Author:     fei.zhang@ga.gov.au
Date:       2016-10-02
"""

import os
import sys
import shutil
from datetime import datetime
import yaml
import logging

from string import Template
# https://www.python.org/dev/peps/pep-0292/
from ConfigParser import ConfigParser
from StringIO import StringIO


logging.basicConfig()
_logger = logging.getLogger(__file__)  # (__name__) ()is root
_logger.setLevel(logging.INFO)
_logger.setLevel(logging.DEBUG)

Satellites = {"LS5": "LANDSAT_5", "LS7": "LANDSAT_7", "LS8": "LANDSAT_8"}


class LoadYamlConf:
    """
    WOfS process initialization setup
    """

    def __init__(self, path2yamlfile):

        self.yamlfile = path2yamlfile

        return

    def write2file(self, configdict, outfile):
        # configdict = {'key1': 'value1', 'key2': 'value2'} nested

        with open(outfile, 'w') as f:
            yaml.dump(configdict, f, default_flow_style=False)

    def loadyaml(self):
        """

        Read data from an input yaml file, which contains run parameters
        :return: a dictionary representation of the yaml file
        """

        with open(self.yamlfile, 'r') as f:
            indict = yaml.safe_load(f)

        # now we got a dictionary: config
        _logger.debug(indict.get('run_id'))

        _logger.debug(yaml.dump(indict, None, default_flow_style=False))  # onto screen

        return indict

    def generate_runid(self):
        """
        generate a run_id
        :return:
        """

        runuser = os.environ['USER']  # _datetime stamp
        dtstamp = datetime.today().isoformat()[:19].replace(':', '-')

        runid = "%s_%s" % (runuser, dtstamp)
        _logger.debug(runid)

        return runid

    

    def get_query_dict(self,config):
        """
        get the query parameters ready for plugin to AGDC-v2 api
        :return:  
        qdict = {
            'time': ('1990-01-01', '1991-01-01'),
            'lat': (-35.2, -35.4),
            'lon': (149.0, 149.2),
        }
        """

        prod_type=config.get('prod_type')
        
        # determine spatial coverage
        lat_range = (
        float(config.get('lat_min_deg')), float(config.get('lat_max_deg')))

        lon_range = (
        float(config.get('lon_min_deg')), float(config.get( 'lon_max_deg')))

        logging.info("Lat range is: %s", lat_range)
        logging.info("Lon range is: %s", lon_range)

        # determine time period

        time_interval = (config.get('start_datetime'), config.get('end_datetime'))
        # dateutil.parser.parse(self.config.get('coverage','start_datetime')), \
        # dateutil.parser.parse(self.config.get('coverage','end_datetime'))  )

        logging.info(str(time_interval))

        # determine satellite list
        #satellites = [Satellites[s] for s in config.get( 'satellites').split(',')]
        #logging.info("Satellites: %s", str(satellites))

        # get a CubeQueryContext (this is a wrapper around the API)

        # cube = DatacubeQueryContext()

        # assemble datasets required by a WOfS run

        # dataset_list = [DatasetType.ARG25, DatasetType.PQ25]

        qdict = {"longitude": lon_range, "latitude": lat_range, "time": time_interval,'prod_type':prod_type}

        return qdict
    
        
    def main(self):
        """ main method to do the setup for a run.

        :return: path2workingdir
        """

        # read in the user input parameter from a yaml file, store as a dict:
        inputdict = self.loadyaml()

        # Sanity-check the user inputs and massage them for subsequent use.
        run_id = inputdict.get('run_id')
        if run_id is None:
            # generate run_id
            run_id = self.generate_runid()
            inputdict['run_id'] = run_id 


        # create the working directory
        basedir_path = inputdict.get('base_dir')
        work_path=os.path.join(basedir_path,run_id)

        if os.path.exists(work_path):
            raise Exception(
                "Error: Directory %s already exists. Please remove it or change your run_id." % (work_path,))
        else:
            # Create a workdir for this wofs run
            os.makedirs(work_path)

        qdict=self.get_query_dict(inputdict)

        print(qdict)
        
        return work_path


#############################################################################
#
# Uasge:  python loadconf.py aoi_dev.yml
#
#############################################################################
if __name__ == "__main__":

    template_client = None
    if len(sys.argv) < 2:
        print "Usage: %s %s" % (sys.argv[0], "path2/aoi.yml")
        sys.exit(1)

    config_infile = sys.argv[1]

    if not os.path.exists(config_infile):
        print ("Error: the input config file %s does not exist"% config_infile)
        sys.exit(2)

    myObj = LoadYamlConf(config_infile)

    workdir = myObj.main()
