#!/usr/bin/env python

############# TODO #####################
# - gerer des parametres optionnels : data_files, includes particuliers
########################################

# This program reads the description.in file and generates all
# the necessary python files for the package
import os,os.path,sys,re,string

# Get the description of the package
execfile('description.py')

# Generic strings for the header and description 
str_header ="""#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 %s.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
#
# $Id$
#------------------------------------------------------------------------------

""" % DESCRIPTION_AUTHOR

str_description ="""\"\"\" A sample description
\"\"\"

"""

# Tokens used to parse a header file to find the classnames 
token_class = [" class \S+ {","class \S+ : (public|protected|private) \S+"]

###############
# header_setup
##############

def header_setup(fh):
    # We first write the header
    fh.write(str_header)

    # And then the import section
    str_import = """# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import glob
from distutils.core import setup, Extension
import distutils.sysconfig
import numpy

"""
    fh.write(str_import)    

def footer_setup(fh,list):
    # fh is a file handler to setup.py
    # list is the list of packages [dana.dir,dana.dir.subdir, ...]
    str_packages = ""
    str_ext =""
    if(len(list) != 0):
        str_packages = str_packages + "'" + list[0] + "'"
        str_ext = str_ext +  re.sub('\.','_',list[0]) + "_ext"
    for i in range(1,len(list)):
        str_packages = str_packages + ",'" + list[i] + "'"
        str_ext = str_ext + "," + re.sub('\.','_',list[i]) + "_ext"
    # We finally write the description of the package
    str_package = """setup (name='%s',
       version = '%s',
       author = '%s',
       author_email = '%s',
       url = '%s',
       description =\"%s\",
       packages = [%s],
       ext_modules = [%s],
       data_files= []
       )""" % ("dana."+DESCRIPTION_NAME, DESCRIPTION_VERSION,
               DESCRIPTION_AUTHOR, DESCRIPTION_EMAIL,
               DESCRIPTION_URL, DESCRIPTION_DESCRIPTION,str_packages,str_ext)
    fh.write(str_package)

def scan_source_file(path):
    # We parse the header file to find the classnames
    # we look for tokens like
    #      class CLASSNAME {
    #      class CLASSNAME : {public,protected,private} MOTHER_CLASSNAME {
    #      class CLASSNAME : {public,protected,private} MOTHER_CLASSNAME     
    filename = string.split(path,"/")[-1]
    file_header = filename[:-3] + ".h"
    path_header = string.replace(path,filename,file_header)
    list_mod = []
    list_mod_detailed = []
    try:
        fin = open(path_header,'r').readlines()
        for line in fin:
            if (re.search(token_class[0],line) != None):
                p = line.split()
                if(len(p) == 3):
                    # Recognized token :     class CLASSNAME {
                    list_mod.append(p[1])
                    list_mod_detailed.append(p[1])
                else:
                    print "[Warning] Unrecognized line %s" % line
            elif (re.search(token_class[1],line) != None):
                p = line.split()
                if(len(p) == 6):
                    # Recognized token :     class CLASSNAME : {public,protected,private} MOTHER_CLASSNAME {
                    #list_mod.append([p[1],p[4]])
                    list_mod.append(p[1])
                    list_mod_detailed.append([p[1],p[4]])
                elif (len(p) == 5):
                    # Recognized token :     class CLASSNAME : {public,protected,private} MOTHER_CLASSNAME
                    list_mod.append(p[1])
                    list_mod_detailed.append([p[1],p[4]])
                else:
                    print "[Warning] Unrecognized line %s" % line    
    except IOError:
        print "[Warning] Cannot open header file %s" % path_header
    return [list_mod,list_mod_detailed]

def write_source_modifications(pkg_name,src_file,fh_modif,list_mod):
    # We suggest in the logfile the modifications to bring to the source files
    # to take into account boost
    fh_modif.write("####################################\n")
    fh_modif.write("Suggested modifications for %s \n" % src_file)
    fh_modif.write("BOOST_PYTHON_MODULE(_%s)"% string.split(pkg_name,".")[-1])
    fh_modif.write("{\n")
    fh_modif.write("using namespace boost::python;\n")
    for m in list_mod:
        if(isinstance(m,list)):
            fh_modif.write("register_ptr_to_python< boost::shared_ptr<%s> >();\n" % m[0])
        else:
            fh_modif.write("register_ptr_to_python< boost::shared_ptr<%s> >();\n" % m)

    for m in list_mod:
        if(isinstance(m,list)):
            fh_modif.write("class_<%s, bases <%s> >(\"%s\",\n" % (m[0],m[1],m[0]))        
            fh_modif.write("\"======================================================================\\n\"\n")
            fh_modif.write("\" \n \" ")
            fh_modif.write("\"A sample description of the module\\n\"\n ")
            fh_modif.write("\"======================================================================\\n\",\n")            
            fh_modif.write("init<> ( \"__init__() -- initializes %s \\n)\"\n" % m[0])
            fh_modif.write(");\n")
        else:
            fh_modif.write("class_<%s>(\"%s\",\n" % (m,m))        
            fh_modif.write("\"======================================================================\\n\"\n")
            fh_modif.write("\" \n \" ")
            fh_modif.write("\"A sample description of the module\\n\"\n ")
            fh_modif.write("\"======================================================================\\n\",\n")            
            fh_modif.write("init<> ( \"__init__() -- initializes %s \\n)\"\n" % m)
            fh_modif.write(");\n")
    fh_modif.write("}\n")   
    fh_modif.write("####################################\n")
    
def rec_generate(pkg_name,fullpath,fh,fh_modif,list):
    # pkg_name is the name of the current package : e.g. dana.dir.subdir
    # fullpath is the fullpath of the directory of the package
    # fh is a file handler for setup.py
    # fh_modif is a file handler for modifications.log
    # list is the list of packages [dana.dir,dana.dir.subdir, ...]
    files = os.listdir (fullpath)
    list_dir = []
    list_modules = []
    has_a_cc = 0 # Does the directory contain a source file ?
    for f in files:
        fullname = os.path.join(fullpath, f)
        if os.path.isdir(fullname):
            # We avoid folders beginning with a "."
            if(re.search("^\.",f) == None):
                # And avoid folders containing a "."
                if(re.search("\.",f) == None):
                    list_dir.append(f)
                else:
                    print "[Fatal] %s : Directory's name cannot contain a \".\" " % fullname
                    print "Aborting..."
                    sys.exit(1)
            else:
                fh_modif.write("Ignoring the directory %s, its name begins by a \".\" \n" % fullname)
        else:
            # We check whether or not the file is a .cc file
            if(re.compile(".cc$").search(fullname) != None):
                list_mod_tmp = scan_source_file(fullname)
                list_modules = list_modules + list_mod_tmp[0]
                write_source_modifications(pkg_name,fullname,fh_modif,list_mod_tmp[1])
                has_a_cc = 1
    
    # We then create the __init__.py file only if the directory contains at least one source file
    if(has_a_cc):
        init_file = os.path.join(fullpath,"__init__.py")
        bool_exit = 1 # It means that we will write a new __init__.py
        #               Its value can be modified by the following test
        if os.path.exists(init_file):
            os.remove(init_file)
            fh_modif.write("Removing %s \n" % init_file)
        # We log the creation of init_file 
        fh_modif.write("Created %s \n" % init_file)
        fout = open(init_file,'w')
        fout.write(str_header)
        fout.write(str_description)
        str_pkg = """from _%s import *
        
""" % string.split(pkg_name,".")[-1]

        for d in list_dir:
            str_pkg = str_pkg + """import %s

""" % d
    
        if(len(list_modules) != 0):
            str_pkg = str_pkg + """__all__ = [\'%s\'""" % list_modules[0]
        elif(len(list_dir) != 0):
            str_pkg = str_pkg + """__all__ = [\'%s\'""" % list_dir[0]
        for i in range(1,len(list_modules)):
            str_pkg = str_pkg + ",'"+list_modules[i]+"'"
        for d in list_dir:
            str_pkg = str_pkg + ",'"+ d + "'"
        if(len(list_modules) != 0 or len(list_dir) != 0 ):
            str_pkg = str_pkg + "]"
        fout.write(str_pkg)

        # We modify the setup.py file
        for d in list_dir:
            sub_pkg_name = pkg_name + "." + d
            list.append(pkg_name + "." + d)
            str_pkg="""
%s_srcs = glob.glob (\"%s/*.cc\")
%s_ext = Extension (
        '%s._%s',
        sources = %s_srcs,
        include_dirs=[numpy.get_include()],
        libraries = ['boost_python', 'boost_thread']
        )

""" % (re.sub('\.','_',sub_pkg_name),re.sub('\.','/',sub_pkg_name),
       re.sub('\.','_',sub_pkg_name),sub_pkg_name,
       d,re.sub('\.','_',sub_pkg_name))
            # Write the block for the package sub_pkg_name in setup.py
            fh.write(str_pkg)
        # And browse all the subdirectories
        for d in list_dir:
            rec_generate(pkg_name + "." + d,os.path.join(fullpath,d),fh,fh_modif,list)
    else:
        fh_modif.write("####################################################\n")
        fh_modif.write("# Ignoring the directory %s and \n" % fullpath)
        fh_modif.write("# all the subdirectories \n")
        fh_modif.write("# It doesn't contain any .cc file\n")
        fh_modif.write("####################################################\n")
        fh_modif.write("# You must suppress manually the entry for the     #\n")
        fh_modif.write("#  package %s in setup.py                          #\n" % pkg_name)
        fh_modif.write("####################################################\n \n")


def root_init(fh_modif):
    if os.path.exists('dana/__init__.py'):
        fh_modif.write("Removing dana/__init__.py \n")
        os.remove('dana/__init__.py')
    fh_modif.write("Created ./dana/__init__.py \n")
    fout = open("dana/__init__.py",'w')
    # We first write the header
    fout.write(str_header)
    fout.write("""\"\"\"Sample description for the package %s

Available packages
------------------
\"\"\"

import os.path, pydoc
    
def _packages_info():
    packagedir = os.path.abspath (os.path.dirname(os.path.realpath(__file__)))
    doc = \"\"
    files = os.listdir (packagedir)
    for f in files:
        fullname = os.path.join(packagedir, f)
        if os.path.isdir(fullname) and not os.path.islink(fullname):
            init_file = os.path.join (fullname, \"__init__.py\")
            if os.path.exists (init_file):
                synopsis = pydoc.synopsis (init_file)
                if synopsis:
                    doc += f.ljust(16,' ') + \"--- \" + synopsis  + \"\\n\"
                else:
                    doc += f.ljust(16,' ') + \"\\t\\t--- N/A\\n\"
    return doc

__doc__ += _packages_info()
__doc__ += \"\\n\\n\"

""" % DESCRIPTION_NAME)    

if __name__ == "__main__":
    list_pkg = []
    if os.path.exists('setup.py'):
        os.remove('setup.py')
    fout_setup = open("setup.py",'w')
    if os.path.exists('modifications.log'):
        os.remove('modifications.log')
    fout_modif = open("modifications.log",'w')
    fout_modif.write("Created ./setup.py \n")
    # We first generate the header of setup.py
    header_setup(fout_setup)
    
    # We then browse the subdirectory dana/ to look for the packages
    packagedir = os.path.abspath (os.path.dirname(os.path.realpath(__file__)))+"/dana"
    files = os.listdir (packagedir)
    for f in files:
        fullname = os.path.join(packagedir, f)
        if os.path.isdir(fullname):
            if(re.compile("^\.").search(f) == None):
                if(re.search("\.",f) == None):
                    # We generate the part for the package f
                    pkg_name = "dana."+f
                    str_pkg="""
%s_srcs = glob.glob (\"%s/*.cc\")
%s_ext = Extension (
        '%s._%s',
        sources = %s_srcs,
        include_dirs=[numpy.get_include()],
        libraries = ['boost_python', 'boost_thread']
        )

""" % (re.sub('\.','_',pkg_name),re.sub('\.','/',pkg_name),
       re.sub('\.','_',pkg_name),pkg_name,
       f,re.sub('\.','_',pkg_name))
                    fout_setup.write(str_pkg)
                    list_pkg.append(pkg_name)
                    # We then recursively do the same job
                    rec_generate(pkg_name,fullname,fout_setup,fout_modif,list_pkg)
                else:
                    print "[Fatal] %s : Directory's name cannot contain a \".\" " % fullname
                    print "Aborting..."
                    sys.exit(1)
            else:
                fout_modif.write("Ignoring the directory %s \n" % fullname)
        else:
            fout_modif.write("Ignoring the file %s \n" % os.path.join(packagedir,f))
    print "Liste des packages : ",list_pkg
    
    # We write the footer of setup.py
    footer_setup(fout_setup,list_pkg)
    
    # We generate the root __init__.py in dana/
    root_init(fout_modif)
