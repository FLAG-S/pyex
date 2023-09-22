import os
import yaml

import subprocess
import re
import copy


class pyex():

    def __init__(self, config, sexpath ='sex', sexfile=None, outdir=None):

        # The path to the SExtractor executable.
        self.sexpath = sexpath

        # The directory in which to store any output files.
        if outdir is not None:
            self.outdir = outdir
            if os.path.isdir(outdir) == False:
                os.makedirs(outdir)
        # If none given, create and store in 'output' directory
        else:
            if os.path.isdir('output') == False:
                os.makedirs('output')
            self.outdir = 'output'

        # Path to a default or custom configuration (.sex) file.
        if sexfile is not None:
            self.sexfile = sexfile
        else:
            self.sexfile = self.generate_default()        

        self.configfile = config
        with open(config, 'r') as file:
            yml = yaml.safe_load_all(file)
            content = []
            for entry in yml:
                content.append(entry)
            self.config, self.measurement = content

        print(f'Initalised a pyex class with: \n Config parameters: {self.config} \n Outputs: {self.measurement} \n')

    def generate_default(self):
        p = subprocess.Popen([self.sexpath, "-d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        f = open('default.sex', 'w')
        f.write(out.decode(encoding='UTF-8'))
        f.close()
        return 'default.sex'
    
    def get_version(self):
        p = subprocess.Popen([self.sexpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        version_match = re.search("[Vv]ersion ([0-9\.])+", err.decode(encoding='UTF-8'))
        if version_match is False:
            raise RuntimeError("Could not determine SExctractor version, check the output of running '%s'" % (self.sexpath))
        version = str(version_match.group()[8:])
        assert len(version) != 0
        return version	
    
    def write_params(self):		

        parameter_filename = f'{self.outdir}/{os.path.splitext(os.path.basename(self.configfile))[0]}.params'
        f = open(parameter_filename, 'w')
        f.write("\n".join(self.measurement))
        f.write("\n")
        f.close()
        return parameter_filename

    def run_SExtractor(self, image, weight = None):
		
        imgconfig = copy.deepcopy(self.config)

        parameter_filename = self.write_params()

		# We set the catalog name :
        if type(image) == list:
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image[1]))[0]}.cat'
        else:
            imgconfig['CATALOG_NAME'] = f'{self.outdir}/{os.path.splitext(os.path.basename(image))[0]}.cat'
        
        # Set the path to the parameter file.
        imgconfig['PARAMETERS_NAME'] = parameter_filename
		
		# We build the command line arguments
        if (type(image) == list) and (type(weight) == list):
            popencmd = [self.sexpath, "-c", self.sexfile, image[0], image[1], '-WEIGHT_IMAGE', f'{weight[0]},{weight[1]}']
        if (type(image) == list) and (type(weight) == type(None)):
            popencmd = [self.sexpath, "-c", self.sexfile, image[0], image[1]]
        if (type(image) == str) and (type(weight) == type(None)):
            popencmd = [self.sexpath, "-c", self.sexfile, image]
        if (type(image) == str) and (type(weight) == str):
            popencmd = [self.sexpath, "-c", self.sexfile, image, '-WEIGHT_IMAGE', weight]

		# We add the current state of config
        for (key, value) in imgconfig.items():
            popencmd.append("-"+str(key))
            popencmd.append(str(value).replace(' ',''))

        p = subprocess.Popen(popencmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        err, out = p.communicate()

        os.remove(parameter_filename)

        print(f'{out.decode(encoding="UTF-8")} \n')

        return