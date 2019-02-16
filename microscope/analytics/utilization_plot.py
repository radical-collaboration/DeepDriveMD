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

# This script plots the core utilization of a set of RP sessions in a stacked
# barplot, where the stacked elements represent how the cores were (or were not)
# utilized.  The elements will always add up to 100%, thus the whole bar
# represents the amount of core-hours available to the session.  We only look at
# events on the agent side.
#
# The bar elements are determined as follows (for each pilot)
#
#  - During bootstrap, configuration, setup, and termination, cores are
#    essentially unused.  We separate out the times needed for those steps, and
#    multiply by pilot size, to derive the number of core hours essentially
#    spent on those activities (see `PILOT_DURATIONS` below).
#
#    NOTE: we assume that the pilot activities thus measured stop at the point
#          when the first unit gets scheduled on a (set of) core(s), and
#          restarts when the last unit gets unscheduled, as that is the time
#          frame where we in principle consider cores to be available to the
#          workload.
#
#  - For each unit, we look at the amount of time that unit has been scheduled
#    on a set of cores, as those cores are then essentially blocked.  Multiplied
#    by the size of the unit, that gives a number of core-hours those cores are
#    'used' for that unit.
#
#    not all of that time is utilized for application use though: some is spent
#    on preparation for execution, on spawning, unscheduling etc.  We separate
#    out those utilizations for each unit.
#
#  - we consider core hours to be additive, in the following sense:
#
#    - all core-hours used by the pilot in various global activities listed in
#      the first point, plus the sumof core hours spend by all units in various
#      individual activities as in the second point, equals the overall core
#      hours available to the pilot.
#
#      This only holds with one caveat: after the agent started to work on unit
#      execution, some cores may not *yet* be allocated (scheduler is too slow),
#      or may not be allocated *anymore* (some units finished, we wait for the
#      remaining ones).  We consider those core-hours as 'idle'.
#
#      Also, the agent itself utilizes one node, and we consider that time as
#      agent utilization overhead.
#
# NOTE: we actually use core-seconds instead of core-hours.  So what?!
#

# ------------------------------------------------------------------------------
#
# absolute utilization: number of core hours per activity
# relative utilization: percentage of total pilot core hours
#
ABSOLUTE = False


# ------------------------------------------------------------------------------
#
# pilot and unit activities: core hours are derived by multiplying the
# respective time durations with pilot size / unit size.  The 'idle' utilization
# and the 'agent' utilization are derived separately.
#
# Note that durations should add up to the `x_total` generations to ensure
# accounting for the complete unit/pilot utilization.
#
PILOT_DURATIONS = {
        'p_total'     : [{STATE: None,            EVENT: 'bootstrap_1_start'},
                         {STATE: None,            EVENT: 'bootstrap_1_stop' }],

        'p_boot'      : [{STATE: None,            EVENT: 'bootstrap_1_start'},
                         {STATE: None,            EVENT: 'sync_rel'         }],
        'p_setup_1'   : [{STATE: None,            EVENT: 'sync_rel'         },
                         {STATE: None,            EVENT: 'orte_dvm_start'   }],
        'p_orte'      : [{STATE: None,            EVENT: 'orte_dvm_start'   },
                         {STATE: None,            EVENT: 'orte_dvm_stop'    }],
        'p_setup_2'   : [{STATE: None,            EVENT: 'orte_dvm_stop'    },
                         {STATE: PMGR_ACTIVE,     EVENT: 'state'            }],
        'p_uexec'     : [{STATE: PMGR_ACTIVE,     EVENT: 'state'            },
                         {STATE: None,            EVENT: 'cmd'              }],
        'p_term'      : [{STATE: None,            EVENT: 'cmd'              },
                         {STATE: None,            EVENT: 'bootstrap_1_stop' }]}

UNIT_DURATIONS = {
        'u_total'     : [{STATE: None,            EVENT: 'schedule_ok'      },
                         {STATE: None,            EVENT: 'unschedule_stop'  }],

        'u_equeue'    : [{STATE: None,            EVENT: 'schedule_ok'      },
                         {STATE: AGENT_EXECUTING, EVENT: 'state'            }],
        'u_eprep'     : [{STATE: AGENT_EXECUTING, EVENT: 'state'            },
                         {STATE: None,            EVENT: 'exec_start'       }],
        'u_exec_rp'   : [{STATE: None,            EVENT: 'exec_start'       },
                         {STATE: None,            EVENT: 'cu_start'         }],
        'u_exec_cu'   : [{STATE: None,            EVENT: 'cu_start'         },
                         {STATE: None,            EVENT: 'cu_exec_start'    }],
        'u_exec_orte' : [{STATE: None,            EVENT: 'cu_exec_start'    },
                         {STATE: None,            EVENT: 'app_start'        }],
        'u_exec_app'  : [{STATE: None,            EVENT: 'app_start'        },
                         {STATE: None,            EVENT: 'app_stop'         }],
        'u_unschedule': [{STATE: None,            EVENT: 'app_stop'         },
                         {STATE: None,            EVENT: 'unschedule_stop'  }]}

DERIVED_DURATIONS = ['p_agent', 'p_idle', 'p_setup']

TRANSLATE_KEYS    = {
                     'p_agent'     : 'agent nodes',
                     'p_boot'      : 'pilot bootstrap',
                     'p_setup'     : 'pilot setup',
                     'p_orte'      : 'orte  setup',
                     'p_term'      : 'pilot termination',
                     'p_idle'      : 'pilot idle',

                     'u_equeue'    : 'CU queued',
                     'u_eprep'     : 'CU preparation',
                     'u_exec_rp'   : 'CU execution (RP)',
                     'u_exec_cu'   : 'CU execution (SYS)',
                     'u_exec_orte' : 'CU execution (ORTE)',
                     'u_exec_app'  : 'CU execution (APP)',
                     'u_unschedule': 'CU unschedule'
                     }

# there must be a better way to do this...
ORDERED_KEYS      = [
                     'p_boot',
                     'p_setup',
                     'p_orte',
                     'u_equeue',
                     'u_eprep',
                     'u_exec_rp',
                     'u_exec_cu',
                     'u_exec_orte',
                     'u_exec_app',
                     'u_unschedule',
                     'p_idle',
                     'p_term',
                     'p_agent',
                     ]

# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    sources     = sys.argv[1:]
    utilization = dict()    # dict of contributions to utilization
    sids        = list()    # used for labels

    if not sources:
        ws_path = 'data/weak_scaling_synapse_titan/optimized'
        ss_path = 'data/strong_scaling_synapse_titan'
        sources = [
                   '%s/rp.session.titan-ext2.itoman.017467.0000' % ss_path,
                   '%s/ws_syn_titan_32_32_1024_60_1.0'           % ws_path,
                   '%s/ws_syn_titan_32_32_1024_60_1.1'           % ws_path,
                   '%s/ws_syn_titan_64_32_2048_60_2.0'           % ws_path,
                   '%s/ws_syn_titan_64_32_2048_60_2.1'           % ws_path,
                   '%s/ws_syn_titan_128_32_4096_60_3.0'          % ws_path,
                   '%s/ws_syn_titan_128_32_4096_60_3.1'          % ws_path,
                   '%s/ws_syn_titan_256_32_8192_60_4.0'          % ws_path,
                   '%s/ws_syn_titan_256_32_8192_60_4.1'          % ws_path,
                   '%s/ws_syn_titan_1024_32_32768_60_6.1'        % ws_path,
                   '%s/ws_syn_titan_2048_32_65536_60_7.0'        % ws_path,
                   '%s/ws_syn_titan_2048_32_65536_60_7.1'        % ws_path,
                   ]


    # create a separate entry in the utilization dict for each source (session)
    for src in sources:

        # always point to the tarballs
        if src[-4:] != '.tbz':
            src += '.tbz'

        print
        print '-----------------------------------------------------------'
        print src

        session = ra.Session(src, 'radical.pilot')
        pilots  = session.filter(etype='pilot', inplace=False)
        units   = session.filter(etype='unit',  inplace=True)
        sid     = session.uid
        sids.append(sid)

        if len(pilots.get()) > 1:
            raise ValueError('Cannot handle multiple pilots')

        print sid

        # compute how many core-hours each duration consumed (or allocated,
        # wasted, etc - depending on the semantic type of duration)
        utilization[sid] = dict()

        for duration in PILOT_DURATIONS:
            utilization[sid][duration] = 0.0

        for duration in UNIT_DURATIONS:
            utilization[sid][duration] = 0.0

        # some additional durations we derive implicitly
        for duration in DERIVED_DURATIONS:
            utilization[sid][duration] = 0.0

        for pilot in pilots.get():

            # we immediately take of the agent nodes, and change pilot_size
            # accordingly
            cpn    = pilot.cfg['cores_per_node']
            anodes = 0
            for agent in pilot.cfg.get('agents'):
                if pilot.cfg['agents'][agent].get('target') == 'node':
                    anodes += 1
            walltime = pilot.duration(event=PILOT_DURATIONS['p_total'])
            psize    = pilot.description['cores']  - anodes * cpn
            utilization[sid]['p_agent'] = walltime * anodes * cpn

            # FIXME: there is something wrong with the above, this triggers
            #        consistency checks
            psize    = pilot.description['cores']
            utilization[sid]['p_agent'] = 0

            # now we can derive the utilization for all other pilot durations
            # specified.  Note that this is now off by some amount for the
            # bootstrapping step where we don't yet have sub-agents, but that
            # can be justified: the sub-agent nodes are explicitly reserved for
            # their purpose at that time. too.
            for duration in PILOT_DURATIONS:
                try:
                    dur = pilot.duration(event=PILOT_DURATIONS[duration])
                except:
                    print 'WARN: miss %s' % duration
                    dur = 0.0
                utilization[sid][duration] += dur * psize

              # if '_2048_' in sid:
              #     print '%-20s: %7.2f' % (duration, dur)


        # we do the same for the unit durations - but here we add up the
        # contributions for all individual units.
        for unit in units.get():
            usize = unit.description['cores']
            for duration in UNIT_DURATIONS:
                dur = unit.duration(event=UNIT_DURATIONS[duration])
                utilization[sid][duration] += dur * usize


        # ----------------------------------------------------------------------
        #
        # sanity checks and derived values
        #
        # we add up 'p_setup_1' and 'p_setup_2' to 'p_setup'
        p_setup_1 = utilization[sid]['p_setup_1'] 
        p_setup_2 = utilization[sid]['p_setup_2'] 
        utilization[sid]['p_setup'] = p_setup_1 + p_setup_2
        del(utilization[sid]['p_setup_1'])
        del(utilization[sid]['p_setup_2'])


        # For both the pilot and the unit utilization, the individual
        # contributions must be the same as the total.
        parts  = 0.0
        tot    = utilization[sid]['p_total']
        print 'tot       : %s' % tot
        for p in utilization[sid]:
            if p != 'p_total' and not p.startswith('u_'):
                parts += utilization[sid][p]
                print '%-10s: %s' % (p, utilization[sid][p])
        print 'pilot: %10.4f - %10.4f = %7.4f' % (tot, parts, tot - parts)
        assert(abs(tot - parts) < 0.0001), '%s == %s' % (tot, parts)

        # same for unit consistency
        parts  = 0.0
        tot    = utilization[sid]['u_total']
        for p in utilization[sid]:
            if p != 'u_total' and not p.startswith('p_'):
                parts += utilization[sid][p]
        print 'unit : %10.4f - %10.4f = %7.4f' % (tot, parts, tot - parts)
        assert(abs(tot - parts) < 0.0001), '%s == %s' % (tot, parts)

        # another sanity check: the pilot `p_uexec` utilization should always be
        # larger than the unit `total`.
        p_uexec = utilization[sid]['p_uexec']
        u_total = utilization[sid]['u_total']
        print 'p_total: %10.4f > %10.4f = %4.1f%%' \
                % (p_uexec, u_total, u_total * 100 / p_uexec)
        assert(p_uexec > u_total), '%s > %s' % (p_uexec, u_total)

        # We in fact know that the difference above, which is not explicitly
        # accounted for otherwise, is attributed to the agent component
        # overhead, and to the DB overhead: its the overhead to get from
        # a functional pilot to the first unit being scheduled, and from the
        # last unit being unscheduled to the pilot being terminated (witing for
        # other units to be finished etc).  We consider that time 'idle'
        utilization[sid]['p_idle' ] = p_uexec - u_total
        del(utilization[sid]['p_uexec'])
        print 'p_idle: %.2f' % utilization[sid]['p_idle']

  # pprint.pprint(utilization)

    # name the individual contributions (so, sans totals).  Also, `p_uexec` was
    # meanwhile replaced by the different unit contributions + `p_idle`.  Also,
    # we have a global `p_setup` now.
    keys  = PILOT_DURATIONS.keys() + UNIT_DURATIONS.keys() + DERIVED_DURATIONS
    keys  = [key for key in keys if 'total'    not in key]
    keys  = [key for key in keys if 'p_uexec'  not in key]
    keys  = [key for key in keys if 'p_setup_' not in key]

    # make sure we can use the ORDERED set if needed.
    assert(len(ORDERED_KEYS) == len(keys))

    ind   = np.arange(len(sids))  # locations for the bars on the x-axis
    width = 0.35                  # width of the bars
    data  = dict()                # the numbers we ultimately plot
    xkeys = list()                # shortened session IDs

    # get the numbers we actually want to plot
    for sid in sids:
        try:
            xkeys.append(sid.split('_')[3])
        except:
            xkeys.append('ss')  # FIXME: derive unit num for stron scaling

        # check that the utilzation contributions add up to the total
        tot_abs = utilization[sid]['p_total']
        tot_rel = 100  # %  <-- shortest comment ever!  <-- not anymore!  <-- :(
        sum_abs = 0
        sum_rel = 0
        for key in keys:
            if key not in data:
                data[key] = list()
            util_abs = utilization[sid][key]
            util_rel = 100.0 * util_abs / tot_abs
            sum_abs += util_abs
            sum_rel += util_rel

            if ABSOLUTE: data[key].append(util_abs)
            else       : data[key].append(util_rel)

        assert(abs(tot_abs - sum_abs) < 0.0001)
        assert(abs(tot_rel - sum_rel) < 0.0001)
      # print 'abs: %10.1f - %10.1f = %4.1f' % (tot_abs, sum_abs, tot_abs-sum_abs)
      # print 'rel: %10.1f - %10.1f = %4.1f' % (tot_rel, sum_rel, tot_rel-sum_rel)

    print
  # pprint.pprint(data)
  # print

    # do the stacked barplots - yes, it is this cumbersome:
    # http://matplotlib.org/examples/pylab_examples/bar_stacked.html
    plt.figure(figsize=(20,14))
    bottom = np.zeros(len(xkeys))
    labels = list()
    plots  = list()
    for key in ORDERED_KEYS:
        plots.append(plt.bar(ind, data[key], width, bottom=bottom))
        bottom += data[key]
        labels.append(TRANSLATE_KEYS[key])

    if ABSOLUTE: plt.ylabel('utilization (% of total resources)')
    else       : plt.ylabel('utilization (in core-seconds)')

    plt.xlabel('number of compute units')
    plt.ylabel('utilization (% of total resources)')
    plt.title ('pilot utilization over workload size (#units)')
    plt.xticks(ind, xkeys)
    handles = [p[0] for p in plots]
    plt.legend(handles, labels, ncol=5, loc='upper left', bbox_to_anchor=(0,1.13))
    plt.savefig('08c_core_utilization.png')
    plt.show()

  # plt.figure(figsize=(20,14))
  # for e_idx in range(len(event_list)):
  #     plt.plot(np_data[:,0], np_data[:,(1+e_idx)],
  #             label='%s - %s' % (event_list[e_idx-1], event_list[e_idx]))
  #
  # plt.yscale('log')
  # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
  #       ncol=2, fancybox=True, shadow=True)
  # plt.savefig('08b_core_utilization.png')
  # plt.show()


# ------------------------------------------------------------------------------

