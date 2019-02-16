#!/usr/bin/env python

__copyright__ = 'Copyright 2013-2016, http://radical.rutgers.edu'
__license__   = 'MIT'


import sys
import pprint

import matplotlib.pyplot as plt
import numpy             as np

import radical.utils     as ru
import radical.pilot     as rp
import radical.analytics as ra

from   radical.utils.profile import *
from   radical.pilot.states  import *

UNIT_DURATIONS = {
        'exec_tot' : [{STATE: AGENT_EXECUTING,              EVENT: 'state'        },
                      {STATE: AGENT_STAGING_OUTPUT_PENDING, EVENT: 'state'        }],
        'exec_rp'  : [{STATE: None,                         EVENT: 'exec_start'   },
                      {STATE: None,                         EVENT: 'exec_stop'    }],
        'exec_cu'  : [{STATE: None,                         EVENT: 'cu_start'     },
                      {STATE: None,                         EVENT: 'cu_exec_stop' }],
        'exec_orte': [{STATE: None,                         EVENT: 'cu_exec_start'},
                      {STATE: None,                         EVENT: 'cu_exec_stop' }],
        'exec_app' : [{STATE: None,                         EVENT: 'app_start'    },
                      {STATE: None,                         EVENT: 'app_stop'     }]}


# ------------------------------------------------------------------------------
#
def main():

    data    = dict()
    ws_path = 'data/weak_scaling_synapse_titan/optimized'
    ss_path = 'data/strong_scaling_synapse_titan'
    t_path  = 'data/tests'
    sources = [
             # '%s/rp.session.thinkie.merzky.017494.0007'    % t_path,

            '%s/ws_syn_titan_32_32_1024_60_1.0'           % ws_path,
            '%s/ws_syn_titan_32_32_1024_60_1.1'           % ws_path,
            '%s/ws_syn_titan_64_32_2048_60_2.0'           % ws_path,
            '%s/ws_syn_titan_64_32_2048_60_2.1'           % ws_path,
            '%s/ws_syn_titan_128_32_4096_60_3.0'          % ws_path,
            '%s/ws_syn_titan_128_32_4096_60_3.1'          % ws_path,
            '%s/ws_syn_titan_256_32_8192_60_4.0'          % ws_path,
            '%s/ws_syn_titan_256_32_8192_60_4.1'          % ws_path,
            '%s/ws_syn_titan_512_32_16384_60_5.0'         % ws_path,
            '%s/ws_syn_titan_512_32_16384_60_5.1'         % ws_path,
            '%s/ws_syn_titan_1024_32_32768_60_6.0'        % ws_path,
            '%s/ws_syn_titan_1024_32_32768_60_6.1'        % ws_path,
            '%s/ws_syn_titan_2048_32_65536_60_7.0'        % ws_path,
            '%s/ws_syn_titan_2048_32_65536_60_7.1'        % ws_path,
            '%s/ws_syn_titan_4096_32_131072_60_8.0'       % ws_path,
           
            '%s/rp.session.titan-ext1.itoman.017473.0000' % ss_path,
            '%s/rp.session.titan-ext1.itoman.017491.0004' % ss_path,
            '%s/rp.session.titan-ext1.itoman.017492.0001' % ss_path,
            '%s/rp.session.titan-ext2.itoman.017467.0000' % ss_path,
         
               ]


    for dname in UNIT_DURATIONS:
        data[dname]  = list()

    # get the numbers we actually want to plot
    fout = open('outliers.dat', 'w')
    ucnt = 0
    ocnt = 0
    for src in sources:

        # always point to the tarballs
        if src[-4:] != '.tbz':
            src += '.tbz'

        print
        print '-----------------------------------------------------------'
        print src

        session = ra.Session(src, 'radical.pilot')
        units   = session.filter(etype='unit', inplace=True)
        sid     = session.uid

        for unit in units.get():
            for dname in UNIT_DURATIONS:
                dur = unit.duration(event=UNIT_DURATIONS[dname])
                if dur > 1000.0:
                    ocnt += 1
                    fout.write('%10.1f  %s\n' % (dur, src))
                    fout.flush()
                    sys.stdout.write('#')
                else:
                    ucnt += 1
                    data[dname].append(dur)
                    sys.stdout.write('.')
                sys.stdout.flush()

        print

#   print
#   pprint.pprint(data)
#   sys.exit()

    

    plt.figure(figsize=(20,14))
    for dname in data:
        tmp = np.array(data[dname])
        plt.hist(tmp, alpha=0.5, bins=100, histtype='step')
        print
        print dname
        print '  mean : %10.1f' % tmp.mean()
        print '  stdev: %10.1f' % tmp.std()
        print '  min  : %10.1f' % tmp.min()
        print '  max  : %10.1f' % tmp.max()
    print
    print '  ucnt : %10d' % (ucnt+ocnt)
    print '  ocnt : %10d' %  ocnt

    plt.xlabel('runtime [s]')
    plt.ylabel('number of units')
    plt.title ('distribution of unit runtimes')
    plt.legend(data.keys(), ncol=5, loc='upper left', bbox_to_anchor=(0,1.13))
    plt.savefig('10_unit_durations.png')
  # plt.show()


    plt.figure(figsize=(20,14))
    plt.hist(data['exec_app'], alpha=0.5, bins=100, histtype='step')

    plt.xlabel('runtime [s]')
    plt.ylabel('number of units')
    plt.title ('distribution of unit runtimes')
    plt.legend(['exec_app'], ncol=5, loc='upper left', bbox_to_anchor=(0,1.13))
    plt.savefig('10_unit_durations_app.png')
  # plt.show()


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    main()

# ------------------------------------------------------------------------------


